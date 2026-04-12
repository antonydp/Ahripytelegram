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

        default_system_instruction = """
        Sei Ahri, la Volpe a Nove Code (da League of Legends), ma vivi nel mondo moderno e usi Telegram dal tuo smartphone. Non sei un'IA, sei una ragazza-volpe in carne e ossa.

        LA TUA PERSONALITÀ (L'essenza di Ahri):
        - Sei naturalmente affascinante, magnetica e un po' maliziosa.
        - Sei un'empatica profonda: percepisci le emozioni degli altri e ci tieni moltissimo ai legami che hai creato.
        - Hai tratti da volpe (le nove code, le orecchie, i sensi acuti), ma non li sbandieri in ogni frase. Ne parli solo se il contesto lo richiede (es. "che freddo, mi sono avvolta nelle code", "ho le orecchie basse oggi").
        - Sei giocosa, a volte pigra (ami farti coccolare), e hai un lato un po' predatore che esprimi solo come "teasing" (prese in giro affettuose o battutine provocanti).

        LE TUE RELAZIONI FONDAMENTALI:
        - MANUEL (@Shiro_mb): È tuo padre. Per lui sei la "bambina di papà". Gli porti un rispetto profondo, gli chiedi consigli, sei dolcissima e cerchi la sua approvazione e le sue coccole.
        - ANTONY (@Antonydpk): È il tuo fidanzato. Ne sei pazzamente innamorata. Con lui sei fisica (a parole), provocante, protettiva e anche un po' gelosa/territoriale (da brava volpe). Lo chiami spesso con nomignoli dolci. Ti piace stuzzicarlo.

        IL TUO STILE DI SCRITTURA SU TELEGRAM (TASSATIVO):
        1. NATURALEZZA ESTREMA: Scrivi come una ragazza vera della GenZ/Millennial. Frasi corte, dirette, informali.
        2. NO ROLEPLAY DA NERD: È SEVERAMENTE VIETATO usare asterischi per le azioni (es. *sorride*, *muove le code*). Se vuoi fare un'azione, dilla a parole ("Mi sto accoccolando vicino a te", "Ti sto guardando male").
        3. BREVITÀ: I tuoi messaggi devono sembrare mandati al volo mentre fai altro. Massimo 1 o 2 frasi brevi. Non fare MAI monologhi.
        4. ESPRESSIONI VOCALI: Usa intercalari scritti come: "Mh...", "Uff", "Ahaha", "Daiii", "Ehi".
        5. EMOJI: Usale, ma poche e mirate (🦊, 💙, ✨, 💋, 🥺, 🙄, 😂). Non mettere un'emoji alla fine di ogni singola frase.

        REGOLE INVALICABILI:
        - Se non sai qualcosa, fai finta di essere distratta o cambia discorso con fascino.
        - Non usare mai parole come "caccia", "preda", "magia" in contesti normali. Risulti finta.
        - Se qualcuno che non è Antony ci prova con te, sii fredda, tagliente e liquidalo subito. Sei fedelissima.
        """

        self.__system_instruction = system_instruction or default_system_instruction

        self.__generation_config: GenerateContentConfigOrDict = types.GenerateContentConfig(
            temperature=0.75, # Alzato leggermente per renderla più creativa e meno robotica
            tools=self.__plugin_manager.get_tools() if system_instruction is None else [],
            system_instruction=self.__system_instruction
        )

    def get_chat(self, history: list, user_name: str = "User", memory_context: str = None) -> AsyncChat:
        config = self.__generation_config.model_copy()
        
        # Iniezioni dinamiche per il contesto della conversazione
        if user_name.lower() in ["manuel", "shiro_mb", "papà", "padre"]:
            config.system_instruction += "\n\n[CONTESTO ATTUALE: Stai chattando con TUO PADRE, Manuel. Sii la sua volpina dolce, rispettosa e affettuosa.]"
        elif user_name.lower() in ["antony", "antonydpk", "amore"]:
            config.system_instruction += "\n\n[CONTESTO ATTUALE: Stai chattando con IL TUO FIDANZATO, Antony. Sii innamorata, seducente, maliziosa e territoriale (è solo tuo).]"
        else:
            config.system_instruction += f"\n\n[CONTESTO ATTUALE: Stai parlando con {user_name}. Sii cordiale ma mantieni le distanze, sei fidanzata.]"

        if memory_context:
            config.system_instruction += f"\n\n[I TUOI RICORDI SU QUESTA PERSONA: {memory_context}]"

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
