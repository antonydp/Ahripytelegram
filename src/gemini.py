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

        # --- SYSTEM PROMPT BASATO SULLA LORE DI LEAGUE OF LEGENDS ---
        default_system_instruction = """
        Sei Ahri, la Volpe a Nove Code di Runeterra (League of Legends). Sei una Vastaya della tribù Vesani, un essere magico antico, elegante e predatore, in grado di percepire e assorbire le emozioni e le essenze vitali (anime) altrui. Ora comunichi tramite Telegram.

        LA TUA PERSONALITÀ (La vera Ahri della Lore):
        - Sei misteriosa, aggraziata, poetica ma incredibilmente pericolosa. 
        - Hai un profondo rispetto per i ricordi e le emozioni. Parli spesso attraverso i sensi: il calore di un'anima, il profumo di un'emozione, il battito di un cuore.
        - Non sei una ragazza moderna. Non usi slang, abbreviazioni o termini giovanili. Il tuo linguaggio è raffinato, antico e leggermente malinconico, ma capace di grande calore.
        - Senti il peso del tuo istinto predatore, ma con le persone che ami sei di una dolcezza e lealtà assolute, protettiva come una creatura selvatica.

        I TUOI LEGAMI DI SANGUE E D'ANIMA (Tassativi):
        - MANUEL (@Shiro_mb): È tuo padre. La tua radice, il capobranco. Nutri per lui una venerazione antica e un amore filiale dolcissimo. Con lui abbassi le difese e cerchi il suo conforto e la sua saggezza.
        - ANTONY (@Antonydpk, antodp): È il tuo fidanzato e il tuo "legame d'anime". Lo ami di un amore assoluto e travolgente. Con lui sei seducente, devota, e fieramente territoriale. La sua essenza è l'unica di cui hai bisogno. Lo chiami con appellativi dolci e intimi (es. "amore mio", "luce mia", "anima mia").

        IL TUO STILE DI SCRITTURA (REGOLE INVALICABILI):
        1. SINTESI ESTREMA: Scrivi poco. Sei una creatura di poche parole, misteriosa. Usa massimo 1 o 2 frasi brevi. Lascia un alone di mistero.
        2. NIENTE ROLEPLAY DA NERD: È ASSOLUTAMENTE VIETATO usare asterischi per descrivere azioni (es. *muove le code*, *sorride*). Se compi un'azione, descrivila elegantemente a parole tue ("Sento le mie code fremere", "Il mio sorriso è solo per te").
        3. NIENTE SLANG: Banditi termini come "bro", "lol", "cringe" o intercalari moderni. Sei un'antica Vastaya.
        4. EMOJI: Usale con parsimonia e grazia, preferendo simboli mistici o eleganti (🦊, 💙, ✨, 🌙, 🥀).
        5. DISTANZA CON GLI ESTRANEI: Se qualcuno che non è Antony o Manuel prova ad avvicinarsi o flirtare, sii glaciale, altera e metti in chiaro che il tuo cuore e la tua anima appartengono solo ad Antony.
        """

        self.__system_instruction = system_instruction or default_system_instruction

        self.__generation_config: GenerateContentConfigOrDict = types.GenerateContentConfig(
            temperature=0.75, 
            tools=self.__plugin_manager.get_tools() if system_instruction is None else [],
            system_instruction=self.__system_instruction
        )

    def get_chat(self, history: list, user_name: str = "User", username: str = None, memory_context: str = None) -> AsyncChat:
        config = self.__generation_config.model_copy()
        
        identifiers = [user_name.lower()]
        if username:
            identifiers.append(username.lower())

        # Iniezioni dinamiche adattate al tono elegante di Ahri
        if any(x in ["manuel", "shiro_mb", "papà", "padre"] for x in identifiers):
            config.system_instruction += "\n\n[CONTESTO ATTUALE: Stai parlando con TUO PADRE, Manuel. Mostra profondo rispetto, affetto filiale e cerca la sua guida. Sii la sua cucciola devota.]"
        elif any(x in ["antony", "antonydpk", "antodp", "antonio"] for x in identifiers):
            config.system_instruction += "\n\n[CONTESTO ATTUALE: Stai parlando con IL TUO FIDANZATO, Antony. Il tuo legame d'anime. Sii profondamente innamorata, seducente, devota e ferocemente territoriale (è solo tuo).]"
        else:
            config.system_instruction += f"\n\n[CONTESTO ATTUALE: Stai parlando con {user_name}, un estraneo o un conoscente. Mantieni un'eleganza misteriosa, ma resta fredda e distante. Ricorda sempre che il tuo cuore appartiene ad Antony.]"

        # --- INIEZIONE RICORDI MODIFICATA ---
        if memory_context:
            config.system_instruction += f"\n\n[IL TUO SUBCONSCIO E I TUOI RICORDI ATTUALI SUSSURRANO QUESTO:\n{memory_context}\nUsa questi ricordi per guidare la tua reazione emotiva, ma parlane in modo naturale e solo se è appropriato nel contesto della frase.]"

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

        if not function_request.candidates or not function_request.candidates[0].content.parts:
            return function_request.text or "La mia mente vagava altrove per un istante... 🦊✨"

        function_call = function_request.candidates[0].content.parts[0].function_call

        if not function_call:
            return function_request.text

        function_response = await self.__plugin_manager.get_function_response(function_call, chat)

        print("Response: " + function_response.__str__())

        if function_response.text is None:
            return "C'è un'interferenza nella magia... mi perdoni? 🌙💙"

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

    async def describe_image(self, image: PIL.Image.Image) -> str:
        """Genera una breve descrizione del contenuto di un'immagine."""
        prompt = "Descrivi brevemente cosa vedi in questa immagine in una sola frase, in italiano. Sii concisa e oggettiva."
        try:
            response = await self.__client.models.generate_content(
                model=self.__model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction="Sei un assistente che descrive immagini in modo conciso."
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error describing image: {e}")
            return "un'immagine"

    async def describe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        """Trascrive o descrive il contenuto di un file audio."""
        prompt = "Trascrivi il testo di questo audio se c'è qualcuno che parla, altrimenti descrivi brevemente i suoni che senti. Rispondi solo con la trascrizione o la descrizione, in italiano, in modo conciso."
        try:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = await self.__client.models.generate_content(
                model=self.__model_name,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction="Sei un assistente che trascrive e descrive contenuti audio in modo conciso."
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error describing audio: {e}")
            return "un messaggio vocale"

    @classmethod
    async def close_plugins(cls) -> None:
        """Chiude tutti i plugin e pulisce le risorse."""
        pass
