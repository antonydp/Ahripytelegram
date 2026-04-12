import os
import time
import random
import re
import anyio
from sqlalchemy import select
from src.entities.chat_message import ChatMessage
from src.entities.diary_entry import DiaryEntry
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, BackgroundTasks
from telegram import Update

from src.chat_service import ChatService
from src.enums import TelegramBotCommands
from src.gemini import Gemini
from src.services.database_service import get_db
from src.services.telegram_service import TelegramService
from os import getenv
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    try:
        telegram_service = TelegramService()
        gemini_chat = Gemini(model_name=getenv('GEMINI_CHAT_MODEL'))
        gemini_decision = Gemini(
            model_name=getenv('GEMINI_DECISION_MODEL'),
            system_instruction="Sei un assistente che decide se Ahri deve rispondere. Rispondi SOLO con un numero (0-100)."
        )

        request.body = await request.json()
        telegram_update = Update.de_json(request.body, telegram_service._telegram_app_bot)
        message_obj = telegram_update.message or telegram_update.edited_message or telegram_update.channel_post
        
        if not message_obj: return 'OK'

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

        if telegram_update.edited_message: return 'OK'
        elif message_obj.text == TelegramBotCommands.START:
            await telegram_service.send_start_message(chat_id=chat_id)
            return 'OK'
        elif message_obj.text == TelegramBotCommands.NEW_CHAT:
            await chat_service.clear_chat_history(db, chat_session.id)
            await telegram_service.send_new_chat_message(chat_id=chat_id)
            return 'OK'
        
        # Filtro duplicati
        tg_msg_date = message_obj.date.replace(tzinfo=timezone.utc) if message_obj.date.tzinfo is None else message_obj.date
        stmt = select(ChatMessage).where(ChatMessage.chat_id == chat_session.id, ChatMessage.role == 'user').order_by(ChatMessage.date.desc()).limit(1)
        res = await db.execute(stmt)
        last_m = res.scalar_one_or_none()
        if last_m:
            db_d = last_m.date.replace(tzinfo=timezone.utc) if last_m.date.tzinfo is None else last_m.date
            if abs((tg_msg_date - db_d).total_seconds()) < 2: return 'OK'

        # Media
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

        # SQL Save
        prompt = f"{user_name}: {raw_text}"
        await chat_service.add_message(db, chat_session.id, prompt, message_obj.date, "user", user_id=user_id, username=username)

        # Decisione Gruppo
        if is_group:
            is_reply = message_obj.reply_to_message and message_obj.reply_to_message.from_user.id == bot_user.id
            is_tag = f"@{bot_user.username}" in (message_obj.text or message_obj.caption or "")
            if not (is_reply or is_tag):
                hist = await chat_service.get_chat_history(db, chat_session.id, limit=5)
                ctx = "\n".join([f"{m['role']}: {m['parts'][0]['text']}" for m in hist])
                ans = await gemini_decision.send_message(f"Contesto:\n{ctx}\nProbabilità risposta (0-100)?", gemini_decision.get_chat([]))
                prob = int(re.search(r'\d+', ans).group()) if re.search(r'\d+', ans) else 50
                if prob < random.randint(40, 70): return 'OK'

        # --- RECUPERO TUTTO IL DIARIO DELL'UTENTE ---
        memory_diary_context = ""
        if user_id:
            res_diary = await db.execute(
                select(DiaryEntry).where(DiaryEntry.user_id == user_id).order_by(DiaryEntry.date.asc())
            )
            entries = res_diary.scalars().all()
            if entries:
                # Uniamo tutte le memorie in un grande blocco di testo
                notes = [f"- {e.memory_text}" for e in entries]
                memory_diary_context = "I TUOI RICORDI SU QUESTO UTENTE:\n" + "\n".join(notes)
        # ----------------------------------------------

        # Chat con Gemini
        full_history = await chat_service.get_chat_history(db, chat_session.id)
        if full_history: full_history.pop() # Togliamo l'ultimo per darlo come prompt

        chat = gemini_chat.get_chat(
            history=full_history,
            user_name=user_name,
            username=username,
            memory_context=memory_diary_context
        )
        await telegram_service._telegram_app_bot.send_chat_action(chat_id=chat_id, action="typing")

        if image: resp = await gemini_chat.send_image(prompt, image, chat, db=db, user_id=user_id)
        elif audio_data: resp = await gemini_chat.send_audio(prompt, audio_data[0], audio_data[1], chat, db=db, user_id=user_id)
        else:
            # Passiamo db e user_id a send_message
            resp = await gemini_chat.send_message(prompt, chat, db=db, user_id=user_id)

        if resp:
            await chat_service.add_message(db, chat_session.id, resp, datetime.now(timezone.utc), "model")
            await telegram_service.send_message(chat_id=chat_id, text=resp, reply_to_message_id=message_obj.message_id)

        return 'OK'
    except Exception as error:
        print(f"Global Error: {error}")
        return {"method": "sendMessage", "chat_id": chat_id, "text": 'C\'è un\'interferenza nella magia... mi perdoni? 🌙💙'}
