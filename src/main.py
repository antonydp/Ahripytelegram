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
from fastapi import Depends, FastAPI, Request, BackgroundTasks
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

# --- FUNZIONE IN BACKGROUND PER MEM0 ---
async def save_memory_background(history_slice, raw_text, full_response, username, user_name, user_id, chat_id, is_group):
    """Esegue l'estrazione e il salvataggio dei ricordi in background senza bloccare la risposta del bot"""
    if not ahri_memory:
        return

    try:
        mem0_messages = []
        for msg in history_slice:
            role = "user" if msg["role"] == "user" else "assistant"
            # Identifica esplicitamente chi sta parlando
            prefix = f"@{username or user_name}: " if role == "user" else "Ahri: "
            mem0_messages.append({"role": role, "content": f"{prefix}{msg['parts'][0]['text']}"})

        mem0_messages.append({"role": "user", "content": f"@{username or user_name}: {raw_text}"})
        mem0_messages.append({"role": "assistant", "content": f"Ahri: {full_response}"})

        extraction_prompt = """
        Agisci come il subconscio emotivo di Ahri (personaggio di League of Legends).
        Il tuo compito è analizzare questa conversazione e aggiornare il tuo 'diario emotivo'.
        REGOLE TASSATIVE:
        1. Ignora convenevoli, saluti, o chiacchiere senza importanza.
        2. Salva SOLO: cambiamenti nei legami affettivi, rabbia, litigi, promesse fatte, segreti confidati, o forti dichiarazioni emotive.
        3. Scrivi il ricordo in modo conciso e in terza persona descrivendo i fatti relazionali (es. 'Ahri ha provato rabbia verso X perché l'ha delusa', 'L'utente Y ha promesso protezione ad Ahri', 'Ahri e Antony hanno riaffermato il loro profondo legame').
        """

        await anyio.to_thread.run_sync(
            lambda: ahri_memory.add(
                messages=mem0_messages,
                agent_id="ahri_bot",
                prompt=extraction_prompt,
                metadata={
                    "username": username or user_name,
                    "original_user_id": str(user_id),
                    "chat_id": str(chat_id),
                    "is_group": is_group
                }
            )
        )
    except Exception as e:
        print(f"Mem0 background add error: {e}")
# ----------------------------------------


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    try:
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
        
        # Controlla duplicati Telegram
        tg_msg_date = message_obj.date
        if tg_msg_date.tzinfo is None:
            tg_msg_date = tg_msg_date.replace(tzinfo=timezone.utc)

        stmt = select(ChatMessage).where(
            ChatMessage.chat_id == chat_session.id,
            ChatMessage.role == 'user'
        ).order_by(ChatMessage.date.desc()).limit(1)

        result = await db.execute(stmt)
        last_user_msg = result.scalar_one_or_none()

        if last_user_msg:
            db_msg_date = last_user_msg.date
            if db_msg_date.tzinfo is None:
                db_msg_date = db_msg_date.replace(tzinfo=timezone.utc)

            if abs((tg_msg_date - db_msg_date).total_seconds()) < 2:
                print("Messaggio duplicato (Retry di Telegram). Ignorato.")
                return 'OK'

        # Identifica e descrivi i media
        image = None
        audio_data = None
        media_description = ""
        caption = (message_obj.text or message_obj.caption or "").replace(f"@{bot_user.username}", "").strip()

        if message_obj.photo:
            image = await telegram_service.get_image_from_message(message_obj)
            if image:
                media_description = await gemini_chat.describe_image(image)
        elif message_obj.voice or message_obj.audio:
            audio_data = await telegram_service.get_audio_from_message(message_obj)
            if audio_data:
                audio_bytes, mime_type = audio_data
                media_description = await gemini_chat.describe_audio(audio_bytes, mime_type)

        if media_description:
            if caption:
                raw_text = f"[{media_description}] {caption}"
            else:
                raw_text = media_description
        else:
            raw_text = caption
            if not raw_text:
                if message_obj.photo:
                    raw_text = "sent an image"
                elif message_obj.voice or message_obj.audio:
                    raw_text = "sent an audio message"

        prompt = f"{user_name}: {raw_text}"

        # Salva in database SQL il messaggio dell'utente
        await chat_service.add_message(
            db, chat_session.id, prompt, message_obj.date, "user",
            user_id=user_id, username=username
        )

        # Logica per ignorare alcuni messaggi se è in gruppo
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

            last_bot_msg = await chat_service.get_last_bot_message(db, chat_session.id)
            if last_bot_msg:
                bot_msg_date = last_bot_msg.date
                if bot_msg_date.tzinfo is None:
                    bot_msg_date = bot_msg_date.replace(tzinfo=timezone.utc)

                diff = (datetime.now(timezone.utc) - bot_msg_date).total_seconds()
                if diff < 25:
                    probability -= 40

            threshold = random.randint(45, 75)
            if probability < threshold:
                return 'OK'

        full_response = ""
        history = await chat_service.get_chat_history(db, chat_session.id)
        if history:
            history.pop()

        # --- MEMORIA CONDIVISA MEM0 (IN LETTURA) ---
        memory_context = ""
        # Cerchiamo solo se il messaggio ha abbastanza testo per evitare query inutili ("ok", "si", etc.)
        if ahri_memory and user_id and (len(raw_text.split()) > 3 or len(raw_text) > 15):
            try:
                search_query = f"Relazioni, segreti, litigi, promesse o emozioni che riguardano l'utente {user_name} (@{username}). Argomento: {raw_text}"

                results = await anyio.to_thread.run_sync(
                    lambda: ahri_memory.search(
                        query=search_query,
                        agent_id="ahri_bot", 
                        limit=4
                    )
                )
                if results:
                    mem_list = results.get("results",[]) if isinstance(results, dict) else results

                    formatted_memories = []
                    for m in mem_list:
                        if isinstance(m, dict) and 'memory' in m:
                            author = m.get("metadata", {}).get("username", "Sconosciuto")
                            formatted_memories.append(f"✦ Ricordo legato a @{author}: {m['memory']}")

                    if formatted_memories:
                        memory_context = "\n".join(formatted_memories)
            except Exception as e:
                print(f"Mem0 search error: {e}")


        chat = gemini_chat.get_chat(
            history=history,
            user_name=user_name,
            username=username,
            memory_context=memory_context
        )
        
        await telegram_service._telegram_app_bot.send_chat_action(chat_id=chat_id, action="typing")

        if image:
            full_response = await gemini_chat.send_image(prompt, image, chat)
        elif audio_data:
            audio_bytes, mime_type = audio_data
            full_response = await gemini_chat.send_audio(prompt, audio_bytes, mime_type, chat)
        else:
            full_response = await gemini_chat.send_message(prompt, chat)

        if full_response:
            # 1. Salva nel DB SQL la risposta (Veloce)
            await chat_service.add_message(db, chat_session.id, full_response, datetime.now(timezone.utc), "model")

            # 2. INVIA SUBITO IL MESSAGGIO ALL'UTENTE
            await telegram_service.send_message(chat_id=chat_id, text=full_response, reply_to_message_id=message_obj.message_id)

            # 3. AVVIA IL SALVATAGGIO IN MEM0 IN BACKGROUND (Lento, ma l'utente non aspetta più)
            if user_id:
                recent_history = history[-4:] if len(history) >= 4 else history
                background_tasks.add_task(
                    save_memory_background,
                    recent_history,
                    raw_text,
                    full_response,
                    username,
                    user_name,
                    user_id,
                    chat_id,
                    is_group
                )

        return 'OK'
    except Exception as error:
        print(f"Error Occurred: {error}")
        return {
            "method": "sendMessage",
            "chat_id": chat_id,
            "text": 'Sorry, I am not able to generate content for you right now. Please try again later. '
        }
