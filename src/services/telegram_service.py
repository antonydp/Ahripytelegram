from collections.abc import Coroutine
from io import BytesIO
from os import getenv
from telegram import Message
from telegram.ext import ApplicationBuilder
from PIL import Image


class TelegramService:
    _telegram_app_bot = None
    _bot_user = None

    def __init__(self):
        self._telegram_app_bot = ApplicationBuilder().token(getenv('TELEGRAM_BOT_TOKEN')).build().bot
        pass

    async def get_me(self):
        if not self._bot_user:
            self._bot_user = await self._telegram_app_bot.get_me()
        return self._bot_user

    def is_secure_webhook_enabled(self) -> bool:
        """Check if secure webhook is enabled.

        Returns:
            True if secure webhook is enabled, False otherwise.
        """
        return getenv("ENABLE_SECURE_WEBHOOK_TOKEN") in ("True", None)
    
    def get_secure_webhook_token(self) -> str:
        """Get the secure webhook token from environment variable.

        Returns:
            The secure webhook token as a string.
        """
        return getenv("TELEGRAM_WEBHOOK_SECRET")
    
    def is_secure_webhook_token_valid(self, headers_token: str) -> bool:
        """Validate the secure webhook token from headers.

        Args:
            headers_token: The token from the request headers.

        Returns:
            True if the token is valid, False otherwise.
        """
        secret_token = self.get_secure_webhook_token()
        return headers_token == secret_token and headers_token is not None
    
    async def send_start_message(self, chat_id: int):
        """Send the start message to the user.

        Args:
            chat_id: The chat ID to send the message to.
        """
        await self.send_message(chat_id=chat_id, text="Welcome to Gemini Bot. Send me a message or an image to get started.")

    async def send_unauthorized_message(self, chat_id: int):
        """Send an unauthorized access message to the user.

        Args:
            chat_id: The chat ID to send the message to.
        """
        await self.send_message(chat_id=chat_id, text="You are not authorized to access this service.")

    async def send_new_chat_message(self, chat_id: int):
        """Send a new chat started message to the user.

        Args:
            chat_id: The chat ID to send the message to.
        """
        await self.send_message(chat_id=chat_id, text="New chat started. How can I assist you?")

    
    async def send_message(self, chat_id: int, text: str, **kwargs) -> Coroutine[Message]:
        """Send a message to the user.

        Args:
            chat_id: The chat ID to send the message to.
            text: The message text content.
        """
        return await self._telegram_app_bot.send_message(chat_id=chat_id, text=text, **kwargs)

    async def send_message_draft(self, chat_id: int, draft_id: int, text: str, **kwargs) -> bool:
        """Send a message draft to the user (streaming).

        Args:
            chat_id: The chat ID to send the message to.
            draft_id: Unique identifier of the message draft.
            text: The message text content.
        """
        return await self._telegram_app_bot.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=text, **kwargs)
    
    async def update_message(self, chat_id: int, message_id: int, text: str, **kwargs) -> Coroutine[Message]:
        """Update a message for the user.

        Args:
            chat_id: The chat ID to update the message for.
            message_id: The message ID to update.
            text: The new message text content.
        """
        return await self._telegram_app_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)
    
    async def get_image_from_message(self, message: Message) -> Image.Image | None:
        """Retrieve the image file bytes from a Telegram message.

        Args:
            message: The Telegram message object.
        Returns:
            The image file bytes if available, None otherwise.
        """

        if message.photo:
            file_id = message.photo[-1].file_id
            file = await self._telegram_app_bot.get_file(file_id)
            bytes_array = await file.download_as_bytearray()
            bytesIO = BytesIO(bytes_array)
            image = Image.open(bytesIO)
            return image
        return None

    async def get_audio_from_message(self, message: Message) -> tuple[bytes, str] | None:
        """Retrieve the audio file bytes and MIME type from a Telegram message.

        Args:
            message: The Telegram message object.
        Returns:
            A tuple of (bytes, mime_type) if available, None otherwise.
        """
        audio_obj = None
        if message.voice:
            audio_obj = message.voice
        elif message.audio:
            audio_obj = message.audio

        if audio_obj:
            file_id = audio_obj.file_id
            mime_type = audio_obj.mime_type or ("audio/ogg" if message.voice else "audio/mpeg")
            file = await self._telegram_app_bot.get_file(file_id)
            bytes_array = await file.download_as_bytearray()
            return bytes(bytes_array), mime_type
        return None