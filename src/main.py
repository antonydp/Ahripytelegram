import os

# Imposta la directory di mem0 in /tmp per evitare errori di Read-only file system su Vercel
os.environ["MEM0_DIR"] = "/tmp/.mem0"

import time
import random
import re
import anyio
from sqlalchemy import select
from src.entities.chat_message import ChatMessage
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from telegram import Update

from src.chat_service import ChatService
from src.enums import TelegramBotCommands
from src.gemini import Gemini
from src.memory_manager import ahri_memory
from src.services.database_service import get_db
from src.services.telegram_service import TelegramService
from os import getenv
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code can go here
    yield
    # Shutdown code can go here

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/webhook")
async def webhook(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        # Understand better how to initialize outside the endpoint
        telegram_service = TelegramService()
        gemini_chat = Gemini(model_name=getenv('GEMINI_CHAT_MODEL'))
        gemini_decision = Gemini(
            model_name=getenv('GEMINI_DECISION_MODEL'),
            system_instruction="Sei un assistente che decide se il bot Ahri deve rispondere a un messaggio. Rispondi SOLO con un numero da 0 a 100."
        )

        request.body = await request.json()

        telegram_update = Update.de_json(request.body, telegram_service._telegram_app_bot)

        message_obj = telegram_update.message or telegram_update.edited_message or telegram_update.channel_post or telegram_update.edited_channel_post
        if not message_obj:
            return 'OK'

        chat_id = message_obj.chat_id
        is_group = message_obj.chat.type in ['group', 'supergroup']

        webhook_secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")

        if(telegram_service.is_secure_webhook_enabled() and telegram_service.is_secure_webhook_token_valid(webhook_secret_token) is False):
            await telegram_service.send_unauthorized_message(chat_id=chat_id)
            return 'OK'
        
        bot_user = await telegram_service.get_me()
        chat_service = ChatService()
        chat_session = await chat_service.get_or_create_session(db, chat_id)

        user_name = message_obj.from_user.first_name if message_obj.from_user else "User"
        user_id = message_obj.from_user.id if message_obj.from_user else None
        username = message_obj.from_user.username if message_obj.from_user else None

        if telegram_update.edited_message:
            return 'OK'
        elif message_obj.text == TelegramBotCommands.START:
            await telegram_service.send_start_message(chat_id=chat_id)
            return 'OK'
        elif message_obj.text == TelegramBotCommands.NEW_CHAT:
            await chat_service.clear_chat_history(db, chat_session.id)
            await telegram_service.send_new_chat_message(chat_id=chat_id)
            return 'OK'
        
        # Construct the user prompt early to save it always
        raw_text = (message_obj.text or message_obj.caption or "").replace(f"@{bot_user.username}", "").strip()
        if message_obj.photo and not message_obj.caption:
            raw_text = "sent an image"
        elif (message_obj.voice or message_obj.audio) and not message_obj.caption:
            raw_text = "sent an audio message"

        prompt = f"{user_name}: {raw_text}"

        # --- INIZIO FIX: CONTROLLO DUPLICATI (RETRY DI TELEGRAM) ---
        stmt = select(ChatMessage).where(
            ChatMessage.chat_id == chat_session.id,
            ChatMessage.role == 'user',
            ChatMessage.text == prompt
        ).order_by(ChatMessage.date.desc()).limit(1)

        result = await db.execute(stmt)
        last_user_msg = result.scalar_one_or_none()

        if last_user_msg:
            msg_date = last_user_msg.date
            if msg_date.tzinfo is None:
                msg_date = msg_date.replace(tzinfo=timezone.utc)

            diff = (datetime.now(timezone.utc) - msg_date).total_seconds()
            # Se lo stesso messaggio è arrivato meno di 30 secondi fa, è un retry di Telegram. Lo ignoriamo.
            if diff < 30:
                print("Messaggio duplicato (Retry di Telegram). Ignorato.")
                return 'OK'
        # --- FINE FIX ---

        # Always save user message to history
        await chat_service.add_message(
            db, chat_session.id, prompt, message_obj.date, "user",
            user_id=user_id, username=username
        )

        # Probability-based decision logic for groups
        if is_group:
            is_reply_to_bot = message_obj.reply_to_message and message_obj.reply_to_message.from_user.id == bot_user.id
            is_mentioned = f"@{bot_user.username}" in (message_obj.text or message_obj.caption or "")

            if is_reply_to_bot or is_mentioned:
                probability = 100
            else:
                recent_history = await chat_service.get_chat_history(db, chat_session.id, limit=6)
                decision_context = "\n".join([f"{m['role']}: {m['parts'][0]['text']}" for m in recent_history])
                decision_prompt = f"Contesto degli ultimi messaggi:\n{decision_context}\n\nQuanto è probabile che Ahri debba rispondere all'ultimo messaggio? (0-100). Rispondi SOLO con il numero."

                decision_chat = gemini_decision.get_chat(history=[])
                decision_response = await gemini_decision.send_message(decision_prompt, decision_chat)

                match = re.search(r'\d+', decision_response)
                probability = int(match.group()) if match else 50

            # Cooldown check
            last_bot_msg = await chat_service.get_last_bot_message(db, chat_session.id)
            if last_bot_msg:
                # Ensure date is timezone-aware for comparison
                bot_msg_date = last_bot_msg.date
                if bot_msg_date.tzinfo is None:
                    bot_msg_date = bot_msg_date.replace(tzinfo=timezone.utc)

                diff = (datetime.now(timezone.utc) - bot_msg_date).total_seconds()
                if diff < 25:
                    probability -= 40

            threshold = random.randint(45, 75)
            if probability < threshold:
                return 'OK'

        # If we reached here, we respond
        full_response = ""
        # Fetch history (it includes the user message we just added)
        history = await chat_service.get_chat_history(db, chat_session.id)
        # Remove the last message from history because send_message will add it again
        if history:
            history.pop()
        # --- MEMORIA CONDIVISA MEM0 ---
        memory_context = ""
        if user_id:
            try:
                search_query = raw_text

                # Se l'utente risponde con una frase corta, cerchiamo di dare contesto alla ricerca
                if len(raw_text.split()) < 13:
                    context_text = ""
                    # Se è una risposta a un messaggio specifico, usiamo quello come contesto
                    if message_obj.reply_to_message:
                        context_text = message_obj.reply_to_message.text or message_obj.reply_to_message.caption or ""
                    # Altrimenti usiamo l'ultimo messaggio della cronologia come fallback
                    elif history:
                        context_text = history[-1]["parts"][0]["text"]

                    if context_text:
                        search_query = f"Contesto: {context_text} - Risposta utente: {raw_text}"

                # cerca i 5 ricordi più rilevanti per questa frase
                results = await anyio.to_thread.run_sync(
                    lambda: ahri_memory.search(
                        query=search_query,
                        user_id=str(user_id),
                        limit=5
                    )
                )
                if results:
                    # In newer mem0ai versions, search returns a dict like {'results': [...]}
                    mem_list = results.get("results",[]) if isinstance(results, dict) else results
                    memory_context = "\n".join([f"- {m['memory']}" for m in mem_list if isinstance(m, dict) and 'memory' in m])
            except Exception as e:
                print(f"Mem0 search error: {e}")


        chat = gemini_chat.get_chat(
            history=history,
            user_name=user_name,
            username=username,
            memory_context=memory_context
        )
        
        # Invia l'azione "Sta scrivendo..." a Telegram per rassicurare l'utente durante l'attesa
        await telegram_service._telegram_app_bot.send_chat_action(chat_id=chat_id, action="typing")

        if message_obj.photo:
            image = await telegram_service.get_image_from_message(message_obj)
            full_response = await gemini_chat.send_image(prompt, image, chat)

        elif message_obj.voice or message_obj.audio:
            audio_data = await telegram_service.get_audio_from_message(message_obj)
            if audio_data:
                audio_bytes, mime_type = audio_data
                full_response = await gemini_chat.send_audio(prompt, audio_bytes, mime_type, chat)
        else:
            # Attendiamo la risposta completa da Gemini (molto più veloce rispetto allo streaming)
            full_response = await gemini_chat.send_message(prompt, chat)

        if full_response:
            await chat_service.add_message(db, chat_session.id, full_response, datetime.now(timezone.utc), "model")

            # --- SALVA IN MEMORIA CONDIVISA (Migliorato) ---
            if user_id:
                try:
                    # Costruiamo una mini-storia degli ultimi 4 messaggi + la risposta attuale
                    mem0_messages = []

                    # Prendiamo gli ultimi 4 messaggi dalla history che abbiamo già estratto
                    recent_history = history[-4:] if len(history) >= 4 else history
                    for msg in recent_history:
                        role = "user" if msg["role"] == "user" else "assistant"
                        mem0_messages.append({"role": role, "content": msg["parts"][0]["text"]})

                    # Aggiungiamo la frase attuale dell'utente e la risposta appena generata
                    mem0_messages.append({"role": "user", "content": raw_text})
                    mem0_messages.append({"role": "assistant", "content": full_response})

                    await anyio.to_thread.run_sync(
                        lambda: ahri_memory.add(
                            messages=mem0_messages, # Ora Mem0 legge l'intero scambio!
                            user_id=str(user_id),
                            metadata={"username": username, "chat_id": str(chat_id), "is_group": is_group}
                        )
                    )
                except Exception as e:
                    print(f"Mem0 add error: {e}")

            # Inviamo il messaggio finale a Telegram in una volta sola
            await telegram_service.send_message(chat_id=chat_id, text=full_response, reply_to_message_id=message_obj.message_id)

        return 'OK'
    except Exception as error:
        print(f"Error Occurred: {error}")
        return {
            "method": "sendMessage",
            "chat_id": chat_id,
            "text": 'Sorry, I am not able to generate content for you right now. Please try again later. '
        }
