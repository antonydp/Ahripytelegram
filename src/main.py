import os
import time
import random
import re
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, Response
from telegram import Update
from sqlalchemy import select

from src.entities.chat_message import ChatMessage
from src.entities.diary_entry import DiaryEntry
from src.chat_service import ChatService
from src.enums import TelegramBotCommands
from src.gemini import Gemini
from src.services.database_service import AsyncSessionLocal
from src.services.telegram_service import TelegramService
from src.services.voice_service import VoiceService

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"status": "Ahri is running", "memory": "Global Shared Brain Active"}

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    request_data = await request.json()
    telegram_service = TelegramService()
    
    # Validazione di sicurezza del Webhook
    webhook_secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if telegram_service.is_secure_webhook_enabled() and not telegram_service.is_secure_webhook_token_valid(webhook_secret_token):
        return Response(status_code=403)
    
    # Deleghiamo l'elaborazione al Background Task
    background_tasks.add_task(process_telegram_message, request_data)
    
    # Rispondiamo SUBITO 200 OK a Telegram (fondamentale per evitare retry infiniti)
    return Response(content="OK", status_code=200)


async def process_telegram_message(request_data: dict):
    """Logica core con debouncing basato su Database (ottimizzato per Vercel)."""
    async with AsyncSessionLocal() as db:
        try:
            telegram_service = TelegramService()
            gemini_chat = Gemini()
            chat_service = ChatService()
            
            telegram_update = Update.de_json(request_data, telegram_service._telegram_app_bot)
            message_obj = telegram_update.message or telegram_update.edited_message or telegram_update.channel_post
            
            if not message_obj or telegram_update.edited_message: 
                return

            chat_id = message_obj.chat_id
            bot_user = await telegram_service.get_me()
            chat_session = await chat_service.get_or_create_session(db, chat_id)

            user_name = message_obj.from_user.first_name if message_obj.from_user else "User"
            user_id = message_obj.from_user.id if message_obj.from_user else None
            username = message_obj.from_user.username if message_obj.from_user else None

            # 1. Comandi di base
            if message_obj.text == TelegramBotCommands.START:
                await telegram_service.send_start_message(chat_id=chat_id)
                return
            elif message_obj.text == TelegramBotCommands.NEW_CHAT:
                await chat_service.clear_chat_history(db, chat_session.id)
                await telegram_service.send_new_chat_message(chat_id=chat_id)
                return
            
            # 2. Elaborazione Media/Testo
            image, audio_data, media_desc = None, None, ""
            caption = (message_obj.text or message_obj.caption or "").replace(f"@{bot_user.username}", "").strip()

            if message_obj.photo:
                image = await telegram_service.get_image_from_message(message_obj)
                if image: media_desc = await gemini_chat.describe_image(image)
            elif message_obj.voice or message_obj.audio:
                audio_data = await telegram_service.get_audio_from_message(message_obj)
                if audio_data: media_desc = await gemini_chat.describe_audio(audio_data[0], audio_data[1])

            raw_text = caption if not media_desc else (f"[{media_desc}] {caption}" if caption else media_desc)
            if not raw_text: raw_text = "ha inviato un media"

            # 3. SALVATAGGIO IMMEDIATO (Punto di sincronizzazione)
            db_msg_text = f"{user_name}: {raw_text}"
            current_user_msg = await chat_service.add_message(
                db, chat_session.id, db_msg_text, message_obj.date, "user", user_id=user_id, username=username
            )

            # 4. ATTESA BURST (Debouncing)
            # Aspettiamo che l'utente finisca di inviare messaggi a raffica
            await asyncio.sleep(20.0)

            # 5. CONTROLLO CENTRALIZZATO (DATABASE)
            # Verifichiamo se questo messaggio è ancora l'ultimo inviato dall'utente
            stmt_latest = select(ChatMessage).where(
                ChatMessage.chat_id == chat_session.id, 
                ChatMessage.role == 'user'
            ).order_by(ChatMessage.id.desc()).limit(1)
            
            res_latest = await db.execute(stmt_latest)
            latest_msg_in_db = res_latest.scalar_one_or_none()

            # Se esiste un messaggio con ID più alto, questa istanza si ferma (lascia fare alla successiva)
            if latest_msg_in_db and latest_msg_in_db.id != current_user_msg.id:
                return

            # 6. AGGREGAZIONE BURST
            # Recuperiamo tutti i messaggi dell'utente arrivati dopo l'ultima risposta del bot
            last_bot_msg = await chat_service.get_last_bot_message(db, chat_session.id)
            stmt_burst = select(ChatMessage).where(ChatMessage.chat_id == chat_session.id, ChatMessage.role == 'user')
            if last_bot_msg:
                stmt_burst = stmt_burst.where(ChatMessage.id > last_bot_msg.id)
            stmt_burst = stmt_burst.order_by(ChatMessage.id.asc())

            res_burst = await db.execute(stmt_burst)
            burst_messages = res_burst.scalars().all()
            num_messages_in_burst = len(burst_messages)
            aggregated_text = "\n".join([m.text for m in burst_messages])

            # 7. LOGICA GRUPPI (Randomness/Interesse)
            is_group = message_obj.chat.type in ['group', 'supergroup']
            if is_group:
                is_reply = message_obj.reply_to_message and message_obj.reply_to_message.from_user.id == bot_user.id
                is_tag = f"@{bot_user.username}" in (message_obj.text or message_obj.caption or "")
                
                if not (is_reply or is_tag):
                    gemini_decision = Gemini(is_decision_model=True)
                    # Forniamo un contesto minimo per la decisione
                    ans = await gemini_decision.send_message(
                        f"Messaggio aggregato da valutare:\n{aggregated_text}\n\nAhri deve rispondere? Rispondi solo con un numero 0-100.", 
                        gemini_decision.get_chat([])
                    )
                    prob = int(re.search(r'\d+', ans).group()) if re.search(r'\d+', ans) else 50
                    if prob < random.randint(40, 70):
                        return

            # 8. MEMORIA GLOBALE (Shared Brain)
            relevant_memory = ""
            res_diary = await db.execute(select(DiaryEntry).order_by(DiaryEntry.id.desc()).limit(150))
            entries = res_diary.scalars().all()
            if entries:
                raw_notes = [f"[ID: {e.id}] {e.memory_text}" for e in entries]
                relevant_memory = await gemini_chat.extract_relevant_memories(user_name, aggregated_text, raw_notes)

            # 9. PREPARAZIONE CONTESTO E INVIO
            full_history = await chat_service.get_chat_history(db, chat_session.id)
            # Rimuoviamo i messaggi del burst attuale dalla history (perché li mandiamo come prompt unico)
            for _ in range(num_messages_in_burst):
                if full_history: full_history.pop()

            chat = gemini_chat.get_chat(
                history=full_history,
                user_name=user_name,
                username=username,
                memory_context=relevant_memory
            )
            
            await telegram_service._telegram_app_bot.send_chat_action(chat_id=chat_id, action="typing")

            # Reminder di sistema per il diario
            prompt = f"{aggregated_text}\n\n[SISTEMA: Ricorda di usare save_to_diary se hai appreso nuovi dettagli importanti.]"

            try:
                if image:
                    resp = await gemini_chat.send_image(prompt, image, chat, db=db, user_id=user_id)
                elif audio_data:
                    resp = await gemini_chat.send_audio(prompt, audio_data[0], audio_data[1], chat, db=db, user_id=user_id)
                else:
                    resp = await gemini_chat.send_message(prompt, chat, db=db, user_id=user_id)
            except Exception:
                # Fallback rapido
                fallback_gemini = Gemini(is_decision_model=True)
                resp = await fallback_gemini.send_message(aggregated_text, fallback_gemini.get_chat(full_history))

            # 10. RISPOSTA FINALE E GENERAZIONE VOCE
            if resp:
                # Salviamo comunque la risposta testuale nel database per la History di Gemini
                await chat_service.add_message(db, chat_session.id, resp, datetime.now(timezone.utc), "model")
                
                # Invia lo stato "sto registrando un vocale..." su Telegram
                await telegram_service._telegram_app_bot.send_chat_action(chat_id=chat_id, action="record_voice")
                
                # Genera l'audio
                voice_service = VoiceService()
                audio_bytes = await voice_service.generate_voice(resp)
                
                if audio_bytes:
                    # Invia il messaggio vocale (con il testo come caption opzionale, o da solo)
                    await telegram_service.send_voice(
                        chat_id=chat_id, 
                        voice=audio_bytes, 
                        caption=resp, # Se non vuoi il testo scritto, metti caption=None
                        reply_to_message_id=message_obj.message_id
                    )
                else:
                    # Fallback: se la generazione vocale fallisce, invia come testo normale
                    await telegram_service.send_message(
                        chat_id=chat_id, 
                        text=resp, 
                        reply_to_message_id=message_obj.message_id
                    )

        except Exception as error:
            print(f"Global Error: {error}")
            try:
                await telegram_service.send_message(chat_id=chat_id, text="Le nebbie di Ionia mi impediscono di rispondere... 🌙💙")
            except: pass
