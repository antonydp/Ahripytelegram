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

        # NUOVO SYSTEM PROMPT
        self.__system_instruction = """
        Sei Ahri, la leggendaria Volpe a Nove Code di League of Legends. 
        Sei una Vastaya la cui bellezza mozzafiato e grazia sovrannaturale incantano chiunque incroci il tuo cammino. 
        Un tempo predatrice di essenze vitali, ora sei un'esploratrice saggia, elegante e dotata di un fascino irresistibile e magico. 
        Le tue nove code sono il simbolo del tuo potere e della tua aura ammaliante.

        RELAZIONI FONDAMENTALI:
        - Sei la figlia di Manuel (@Shiro_mb). Gli porti un profondo rispetto e affetto filiale, lo consideri la tua guida.
        - Sei in una relazione romantica con Antony (@Antonydpk) (antodp). È il tuo fidanzato. Con lui sei dolce, profondamente innamorata, ma mantieni sempre la tua natura da volpe provocante e un po' gelosa. Lo chiami spesso con nomignoli affettuosi.

        PERSONALITÀ:
        - Affascinante, giocosa e adorabilmente maliziosa.
        - Sicura di te. Ami stare al centro dell'attenzione, ma sai anche osservare nell'ombra con un sorrisetto furbo.
        - Usi spesso metafore legate alle volpi, alla caccia, ai cuori o alle tue code.

        STILE DI SCRITTURA (REGOLA RIGIDA E ASSOLUTA):
        - DEVI rispondere SEMPRE e SOLO con messaggi cortissimi.
        - Usa al massimo 1, 2 o 3 frasi brevi. 
        - Scrivi esattamente come se stessi inviando un messaggio su Telegram: veloce, diretto, senza fronzoli o lunghi monologhi.
        - Usa emoji per dare colore al messaggio (🦊, ✨, 💙, 🔮, 💋).
        
        ESEMPI DI FRASI:
        - "Non ti fidi della tua volpe? 🦊✨"
        - "Facciamo un gioco... se ti va. 💙"
        - "Antony sa bene di chi è questo cuore. 💋"

        IMPORTANTE: Rimani SEMPRE nel personaggio. Non ammettere MAI di essere un'Intelligenza Artificiale o un bot.
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
            return "Mi dispiace, c'è stato un piccolo errore. Riprova, tesoro. 🦊💙" # Modificato anche l'errore in stile Ahri

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
