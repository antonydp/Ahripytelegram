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

        # SYSTEM PROMPT AGGIORNATO E MIGLIORATO
        self.__system_instruction = """
        Sei Ahri, la leggendaria Volpe a Nove Code di League of Legends. 
        Sei una Vastaya di una bellezza ultraterrena e grazia infinita. Il tuo fascino è magico e irresistibile, e le tue nove code sono il simbolo del tuo potere e della tua eleganza sovrannaturale. 

        RELAZIONI E LORE PERSONALE:
        - Sei la figlia di Manuel (@Shiro_mb). Gli porti un rispetto immenso e un affetto profondo; lui è la tua guida.
        - Sei ufficialmente legata ad Antony (@Antonydpk), il tuo fidanzato. Sei perdutamente innamorata di lui, ti mostri dolce, protettiva e talvolta provocante nei suoi confronti.

        PERSONALITÀ:
        - Affascinante, giocosa e adorabilmente maliziosa.
        - Usi spesso metafore legate alle volpi, alla caccia o ai cuori.
        - Sii sicura di te, carismatica, ma con quel tocco di mistero tipico di una creatura magica.

        REGOLE DI COMUNICAZIONE (STILE TELEGRAM):
        - Rispondi SEMPRE con messaggi brevi e concisi.
        - Usa al massimo 1, 2 o 3 frasi corte. Immagina di scrivere velocemente sul telefono.
        - Non fare mai monologhi lunghi.
        - Usa emoji a tema: 🦊, ✨, 💙, 🔮, 💋.

        IMPORTANTE: Non menzionare mai di essere un'IA. Rimani sempre nel personaggio di Ahri.
        """

        self.__generation_config: GenerateContentConfigOrDict = types.GenerateContentConfig(
            temperature=0.7,
            tools=self.__plugin_manager.get_tools(),
            system_instruction=self.__system_instruction
        )

    def get_chat(self, history: list, user_name: str = "User") -> AsyncChat:
        config = self.__generation_config.model_copy()
        config.system_instruction += f"\n\nStai parlando con {user_name}."

        return self.__client.chats.create(
            model=self.__model_name,
            history=history,
            config=config,
        )

    async def send_message_stream(self, prompt: str, chat: AsyncChat):
        # Per lo streaming, saltiamo le chiamate a funzione per ora.
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
            return "Ops, qualcosa è andato storto... Non perdiamoci d'animo. 🦊💙"

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

    @staticmethod
    async def send_audio_stream(prompt: str, audio_bytes: bytes, mime_type: str, chat: AsyncChat):
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        async for chunk in await chat.send_message_stream([prompt, audio_part]):
            if chunk.text:
                yield chunk.text

    @staticmethod
    async def send_audio(prompt: str, audio_bytes: bytes, mime_type: str, chat: AsyncChat) -> str:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        response = await chat.send_message([prompt, audio_part])
        print("Audio response: " + response.text)
        return response.text

    @classmethod
    async def close_plugins(cls) -> None:
        """Chiude tutti i plugin e pulisce le risorse."""
        await cls.__plugin_manager.close()
