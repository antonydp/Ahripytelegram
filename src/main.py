import os
import time
import random
import re
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, BackgroundTasks, Response
from telegram import Update
from sqlalchemy import select

from src.entities.chat_message import ChatMessage
from src.entities.diary_entry import DiaryEntry
from src.chat_service import ChatService
from src.enums import TelegramBotCommands
from src.gemini import Gemini
from src.services.database_service import AsyncSessionLocal
from src.services.telegram_service import TelegramService

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
    
    # Rispondiamo SUBITO 200 OK a Telegram (evita i timeout di 15 secondi)
    return Response(content="OK", status_code=200)


async def process_telegram_message(request_data: dict):
    """Logica core del bot eseguita in background."""
    async with AsyncSessionLocal() as db:
        try:
            telegram_service = TelegramService()
            gemini_chat = Gemini()
            
            telegram_update = Update.de_json(request_data, telegram_service._telegram_app_bot)
            message_obj = telegram_update.message or telegram_update.edited_message or telegram_update.channel_post
            
            if not message_obj or telegram_update.edited_message: 
                return

            chat_id = message_obj.chat_id
            is_group = message_obj.chat.type in ['group', 'supergroup']
            bot_user = await telegram_service.get_me()
            chat_service = ChatService()
            chat_session = await chat_service.get_or_create_session(db, chat_id)

            user_name = message_obj.from_user.first_name if message_obj.from_user else "User"
            user_id = message_obj.from_user.id if message_obj.from_user else None
            username = message_obj.from_user.username if message_obj.from_user else None

            # Comandi di base
            if message_obj.text == TelegramBotCommands.START:
                await telegram_service.send_start_message(chat_id=chat_id)
                return
            elif message_obj.text == TelegramBotCommands.NEW_CHAT:
                await chat_service.clear_chat_history(db, chat_session.id)
                await telegram_service.send_new_chat_message(chat_id=chat_id)
                return
            
            # Filtro anti-duplicati (se Telegram dovesse comunque forzare un reinvio)
            tg_msg_date = message_obj.date.replace(tzinfo=timezone.utc) if message_obj.date.tzinfo is None else message_obj.date
            stmt = select(ChatMessage).where(ChatMessage.chat_id == chat_session.id, ChatMessage.role == 'user').order_by(ChatMessage.date.desc()).limit(1)
            res = await db.execute(stmt)
            last_m = res.scalar_one_or_none()
            if last_m:
                db_d = last_m.date.replace(tzinfo=timezone.utc) if last_m.date.tzinfo is None else last_m.date
                if abs((tg_msg_date - db_d).total_seconds()) < 2: 
                    return

            # Elaborazione Media
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

            # Salvataggio messaggio utente
            prompt = f"{user_name}: {raw_text}\n\n[SISTEMA: Considera attentamente se quanto appena detto merita di essere salvato nel diario o se richiede l'aggiornamento di un ricordo esistente tramite ID.]"
            await chat_service.add_message(db, chat_session.id, prompt, message_obj.date, "user", user_id=user_id, username=username)

            # Decisione per i Gruppi (Randomness)
            if is_group:
                is_reply = message_obj.reply_to_message and message_obj.reply_to_message.from_user.id == bot_user.id
                is_tag = f"@{bot_user.username}" in (message_obj.text or message_obj.caption or "")
                if not (is_reply or is_tag):
                    gemini_decision = Gemini(system_instruction="Sei un bot router. Rispondi SOLO con un numero (0-100) indicando la probabilità che Ahri debba rispondere a questo contesto.")
                    hist = await chat_service.get_chat_history(db, chat_session.id, limit=5)
                    ctx = "\n".join([f"{m['role']}: {m['parts'][0]['text']}" for m in hist])
                    ans = await gemini_decision.send_message(f"Contesto:\n{ctx}\nProbabilità risposta (0-100)?", gemini_decision.get_chat([]))
                    prob = int(re.search(r'\d+', ans).group()) if re.search(r'\d+', ans) else 50
                    if prob < random.randint(40, 70): 
                        return

            # ==========================================
            # MEMORIA GLOBALE INTELLIGENTE (SHARED BRAIN)
            # ==========================================
            relevant_memory = ""
            
            # Recuperiamo le ultime 200 memorie GLOBALI (di tutti gli utenti)
            res_diary = await db.execute(
                select(DiaryEntry).order_by(DiaryEntry.date.desc()).limit(200)
            )
            entries = res_diary.scalars().all()
            
            if entries:
                raw_notes = [f"[ID: {e.id}] {e.memory_text}" for e in entries]
                # Estraiamo solo i segreti/ricordi pertinenti a QUESTO utente o a chi viene nominato
                relevant_memory = await gemini_chat.extract_relevant_memories(user_name, raw_text, raw_notes)

            # Preparazione Chat History per Ahri
            full_history = await chat_service.get_chat_history(db, chat_session.id)
            if full_history: full_history.pop() # Rimuoviamo l'ultimo perché lo passiamo come prompt

            # Creazione della chat (Iniezione STM + LTM)
            chat = gemini_chat.get_chat(
                history=full_history,
                user_name=user_name,
                username=username,
                memory_context=relevant_memory
            )
            
            await telegram_service._telegram_app_bot.send_chat_action(chat_id=chat_id, action="typing")

            # Invio Messaggio a Gemini
            if image: 
                resp = await gemini_chat.send_image(prompt, image, chat, db=db, user_id=user_id)
            elif audio_data: 
                resp = await gemini_chat.send_audio(prompt, audio_data[0], audio_data[1], chat, db=db, user_id=user_id)
            else: 
                resp = await gemini_chat.send_message(prompt, chat, db=db, user_id=user_id)

            # Salvataggio e Invio risposta
            if resp:
                await chat_service.add_message(db, chat_session.id, resp, datetime.now(timezone.utc), "model")
                await telegram_service.send_message(chat_id=chat_id, text=resp, reply_to_message_id=message_obj.message_id)

        except Exception as error:
            print(f"Global Error processing task: {error}")
            try:
                await telegram_service.send_message(chat_id=chat_id, text="C'è un'interferenza nella magia... mi perdoni? 🌙💙")
            except: 
                pass
