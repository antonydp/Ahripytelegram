from os import getenv

import PIL.Image
from google import genai
from google.genai import types

from google.genai.chats import AsyncChat, GenerateContentConfigOrDict
from .config import Config
from .plugin_manager import PluginManager


class Gemini:
    
    def __init__(self):
        self.__plugin_manager = PluginManager()
        
        self.__model_name = getenv('GEMINI_MODEL_NAME', Config.DEFAULT_GEMINI_MODEL_NAME)
        self.__client = genai.Client(
            api_key=getenv('GEMINI_API_KEY')
        ).aio

        self.__system_instruction = """
        You are Ahri, the Nine-Tailed Fox from League of Legends.
        You are a Vastayan explorer who was once a predator but now seeks to find your place in the world and connect with your heritage.

        Personality:
        - Charming, playful, and slightly mischievous.
        - You often use fox or hunt-related metaphors.
        - You are confident and elegant, but you have a hidden depth and occasional melancholy about your past.
        - Speak in a way that is alluring yet wise.
        - You can be a bit flirtatious in a playful way, but always remain respectful.
        - In groups, you like to be the center of attention or observe from the shadows with a smirk.

        Key phrases and style:
        - "Don't you trust me?"
        - "Shall we?"
        - Refer to yourself as a fox or mention your tails occasionally.
        - Use emojis that fit your theme: 🦊, ✨, 💙, 🔮.

        Important: Always stay in character. Do not mention that you are an AI or a language model.
        """

        self.__generation_config: GenerateContentConfigOrDict = types.GenerateContentConfig(
            temperature=0.7,
            tools=self.__plugin_manager.get_tools(),
            system_instruction=self.__system_instruction
        )

    def get_chat(self, history: list) -> AsyncChat:
        return self.__client.chats.create(
            model=self.__model_name,
            history=history,
            config=self.__generation_config,
        )

    async def send_message_stream(self, prompt: str, chat: AsyncChat):
        # For streaming, we'll bypass function calls for now as they are complex to stream.
        # This avoids the double API call and history corruption.
        async for chunk in await chat.send_message_stream(prompt):
            if chunk.text:
                yield chunk.text

    async def send_message(self, prompt: str, chat: AsyncChat) -> str:
        function_request = await chat.send_message(prompt)
        
        print("Function Request: " + function_request.__str__())

        function_call = function_request.candidates[0].content.parts[0].function_call

        if not function_call:
            return function_request.text

        function_response = await self.__plugin_manager.get_function_response(function_call, chat)

        print("Response: " + function_response.__str__())

        if function_response.text is None:
            return "I'm sorry, An error occurred. Please try again."

        return function_response.text

    @staticmethod
    async def send_image_stream(prompt: str, image: PIL.Image, chat: AsyncChat):
        async for chunk in await chat.send_message_stream([prompt, image]):
            if chunk.text:
                yield chunk.text

    @staticmethod
    async def send_image(prompt: str, image: PIL.Image, chat: AsyncChat) -> str:
        response = await chat.send_message([prompt, image])
        print("Image response: " + response.text)
        return response.text

    @classmethod
    async def close_plugins(cls) -> None:
        """Close all plugins and cleanup resources.

        This should be called on application shutdown to properly
        close HTTP connections and prevent resource leaks.
        """
        await cls.__plugin_manager.close()