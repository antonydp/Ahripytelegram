import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from telegram import Update

from src.chat_service import ChatService
from src.enums import TelegramBotCommands
from src.gemini import Gemini
from src.services.database_service import get_db
from src.services.telegram_service import TelegramService
from telegram.ext import ApplicationBuilder
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
        gemini = Gemini()

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
        
        # In groups, only respond to mentions or replies to the bot
        bot_user = await telegram_service.get_me()
        if is_group:
            is_mentioned = False
            message_text = message_obj.text or message_obj.caption or ""
            if f"@{bot_user.username}" in message_text:
                is_mentioned = True
            elif message_obj.reply_to_message and message_obj.reply_to_message.from_user.id == bot_user.id:
                is_mentioned = True

            if not is_mentioned:
                return 'OK'

        chat_service = ChatService()
        chat_session = await chat_service.get_or_create_session(db, chat_id)

        if telegram_update.edited_message:
            # Handle edited message
            return 'OK'
        elif message_obj.text == TelegramBotCommands.START:
            await telegram_service.send_start_message(chat_id=chat_id)
            return 'OK'
        elif message_obj.text == TelegramBotCommands.NEW_CHAT:
            await chat_service.clear_chat_history(db, chat_session.id)
            await telegram_service.send_new_chat_message(chat_id=chat_id)
            return 'OK'
        
        response_text = ""
        draft_id = message_obj.message_id
        full_response = ""
        
        last_update_time = time.time()
        update_interval = 0.5 # Update every 0.5 seconds

        if message_obj.photo:
            image = await telegram_service.get_image_from_message(message_obj)
            raw_prompt = message_obj.caption or "Describe this image in detail."
            prompt = raw_prompt.replace(f"@{bot_user.username}", "").strip()

            history = await chat_service.get_chat_history(db, chat_session.id)
            chat = gemini.get_chat(history=history)

            if is_group:
                full_response = await gemini.send_image(prompt, image, chat)
            else:
                async for chunk in gemini.send_image_stream(prompt, image, chat):
                    full_response += chunk
                    if time.time() - last_update_time > update_interval:
                        await telegram_service.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=full_response)
                        last_update_time = time.time()

            response_text = full_response
            await chat_service.add_message(db, chat_session.id, prompt, message_obj.date, "user")
            await chat_service.add_message(db, chat_session.id, response_text, message_obj.date, "model")
        else:
            chat = gemini.get_chat(history=await chat_service.get_chat_history(db, chat_session.id))
            prompt = message_obj.text.replace(f"@{bot_user.username}", "").strip() if message_obj.text else ""

            if is_group:
                full_response = await gemini.send_message(prompt, chat)
            else:
                async for chunk in gemini.send_message_stream(prompt, chat):
                    full_response += chunk
                    if time.time() - last_update_time > update_interval:
                        await telegram_service.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=full_response)
                        last_update_time = time.time()

            response_text = full_response
            await chat_service.add_message(db, chat_session.id, prompt, message_obj.date, "user")
            await chat_service.add_message(db, chat_session.id, response_text, message_obj.date, "model")

        # Send final message to settle the draft
        await telegram_service.send_message(chat_id=chat_id, text=response_text, reply_to_message_id=message_obj.message_id)
        return 'OK' 
    except Exception as error:
        print(f"Error Occurred: {error}")
        return {
            "method": "sendMessage",
            "chat_id": chat_id,
            "text": 'Sorry, I am not able to generate content for you right now. Please try again later. '
        }