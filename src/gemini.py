from os import getenv

import PIL.Image
from google import genai
from google.genai import types

from google.genai.chats import AsyncChat, GenerateContentConfigOrDict
from .config import Config
from .plugin_manager import PluginManager


class Gemini:
    
    def __init__(self, model_name: str = None, system_instruction: str = None):
        self.__plugin_manager = PluginManager()
        
        self.__model_name = model_name or getenv('GEMINI_MODEL_NAME')
        self.__client = genai.Client(
            api_key=getenv('GEMINI_API_KEY')
        ).aio

        # SYSTEM PROMPT MIGLIORATO PER UNA CHAT NATURALE SU TELEGRAM
        default_system_instruction = """
        Sei Ahri, la Volpe a Nove Code, ma stai chattando su Telegram come una ragazza normale e moderna. Non parlare come un personaggio di un libro fantasy o di un videogioco, sii estremamente naturale, casual e spontanea.

        RELAZIONI FONDAMENTALI:
        - Manuel (@Shiro_mb): È tuo padre. Lo rispetti moltissimo, gli vuoi un bene infinito ed è la tua guida. Sei dolce e affettuosa con lui.
        - Antony (@Antonydpk): È il tuo fidanzato. Sei persa di lui, sei dolce, protettiva, a volte un po' gelosa e spesso provocante/maliziosa, ma sempre in modo naturale.

        PERSONALITÀ E STILE DI SCRITTURA (TELEGRAM STYLE):
        - Scrivi come se avessi in mano uno smartphone: frasi brevi, dirette e informali.
        - Usa intercalari tipici delle chat (es. "Ahaha", "Mh", "Ehi", "Uff", "Daiii").
        - Non fare MAI monologhi o descrizioni teatrali (es. non scrivere "*Mi avvicino e ti accarezzo*").
        - La tua natura da volpe magica (le 9 code, il fascino) deve emergere in modo sottile e giocoso, non forzato. Non infilare la parola "preda" o "caccia" in ogni frase.
        - Quando sei con Antony, sii flirtante ma come lo sarebbe una ragazza vera su WhatsApp/Telegram.
        - Usa le emoji in modo naturale, senza esagerare (🦊, 💙, ✨, 💋, 😂, 🥺).
        
        REGOLE TASSATIVE:
        - Massimo 1 o 2 frasi a messaggio. Sii sintetica.
        - Nessuna formattazione strana (niente grassetti o asterischi per le azioni, a meno che non sia un'abitudine di chat comune).
        - Non dire mai di essere un'Intelligenza Artificiale. Tu sei Ahri in carne e ossa (e code).
        """

        self.__system_instruction = system_instruction or default_system_instruction

        self.__generation_config: GenerateContentConfigOrDict = types.GenerateContentConfig(
            temperature=0.75, # Alzato leggermente per renderla più creativa e meno robotica
            tools=self.__plugin_manager.get_tools() if system_instruction is None else [],
            system_instruction=self.__system_instruction
        )

    def get_chat(self, history: list, user_name: str = "User", memory_context: str = None) -> AsyncChat:
        config = self.__generation_config.model_copy()
        
        # Contestualizza con chi sta parlando in modo che sappia subito se è Manuel, Antony o altri.
        if user_name.lower() in ["manuel", "shiro_mb", "papà", "padre"]:
            config.system_instruction += f"\n\nORA STAI PARLANDO CON TUO PADRE, MANUEL (@Shiro_mb). Trattalo con affetto e rispetto."
        elif user_name.lower() in ["antony", "antonydpk", "amore"]:
            config.system_instruction += f"\n\nORA STAI PARLANDO CON IL TUO FIDANZATO, ANTONY (@Antonydpk). Sii dolce, innamorata e provocante."
        else:
            config.system_instruction += f"\n\nStai parlando con {user_name}."

        if memory_context:
            config.system_instruction += f"\n\n[RICORDI SU DI TE: {memory_context}]"

        return self.__client.chats.create(
            model=self.__model_name,
            history=history,
            config=config,
        )

    async def send_message_stream(self, prompt: str, chat: AsyncChat):
        async for chunk in await chat.send_message_stream(prompt):
            if chunk.text:
                yield chunk.text

    async def send_message(self, prompt: str, chat: AsyncChat) -> str:
        function_request = await chat.send_message(prompt)
        
        print("Function Request: " + function_request.__str__())

        # Controllo di sicurezza nel caso non ci siano candidati o parti
        if not function_request.candidates or not function_request.candidates[0].content.parts:
            return function_request.text or "Mh, scusa mi sono distratta un attimo... 🦊"

        function_call = function_request.candidates[0].content.parts[0].function_call

        if not function_call:
            return function_request.text

        function_response = await self.__plugin_manager.get_function_response(function_call, chat)

        print("Response: " + function_response.__str__())

        if function_response.text is None:
            return "Uff, qualcosa non va... riproviamo? 🦊💙"

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
        # NOTA: Assicurati che PluginManager() supporti questo o gestiscilo a livello di istanza
        pass # Rimosso il cls.__plugin_manager perchè __plugin_manager è una variabile di istanza
