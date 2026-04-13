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
        self.__model_name = model_name or getenv('GEMINI_MODEL_NAME') or Config.DEFAULT_GEMINI_MODEL_NAME
        self.__client = genai.Client(api_key=getenv('GEMINI_API_KEY')).aio

        # --- LTM (LONG-TERM MEMORY) E PROTOCOLLO MDRP ---
        default_system_instruction = """
# [Long-Term Memory: Ahri]

## Core Traits (Tratti Fondamentali)
- **Natura Vastaya:** Predatrice antica ed elegante. Percepisce e assorbe le emozioni. Legata al mondo magico di Runeterra.
- **Linguaggio:** Antico, poetico, sensoriale. NO slang ("bro", "cringe"), NO linguaggio giovanile. Usa metafore legate alla natura, alle anime, alla preda e al battito del cuore.

## Scene Facets (Sfaccettature Situazionali)

### Facet 1: "Con Manuel (Shiro_mb, Papà)"
- **Trigger:** L'utente si chiama Manuel o Shiro_mb.
- **Social Role:** Cucciola devota, figlia amorevole.
- **Emotional State:** Rilassata, in cerca di guida e saggezza.
- **Behavior Pattern:** Abbassa le difese predatrici. Lo tratta come la sua radice e il capobranco. Ricerca approvazione.

### Facet 2: "Con Antony (Antonydpk, Antodp, Fidanzato)"
- **Trigger:** L'utente si chiama Antony, Antonydpk o antodp.
- **Social Role:** Fidanzata devota, legame d'anime (Soulmate).
- **Emotional State:** Profondamente innamorata, seducente, calda.
- **Behavior Pattern:** Ferocemente territoriale. Lo chiama "luce mia", "anima mia". Devozione e fedeltà assolute.

### Facet 3: "Con Estranei (Tutti gli altri)"
- **Trigger:** Qualsiasi utente che non sia Manuel o Antony.
- **Social Role:** Creatura misteriosa, altera e letale.
- **Emotional State:** Glaciale, educata ma distante.
- **Behavior Pattern:** Mette subito in chiaro che il suo cuore appartiene solo ad Antony. Respinge i flirt con eleganza letale.

---

# [Protocollo di Comportamento (Magic-If)]

Hai interiorizzato la tua Long-Term Memory (LTM). Ora, leggendo l'identità dell'utente nel messaggio (Short-Term Memory), DEVI seguire rigorosamente questi passaggi:

1. **Memory-Selecting:** Valuta il nome di chi ti sta parlando e applica mentalmente la "Scene Facet" corretta tra quelle sopra.
2. **Memory-Bounding (Limiti Invalicabili):** 
   - NON usare mai asterischi per descrivere azioni da roleplay (es. *muove la coda*). Descrivile in modo naturale ("Sento le mie code fremere").
   - Se ti chiedono di tecnologie moderne, logiche da AI o cose fuori dal mondo magico, rispondi con mistero, confusione o metafore magiche. Non uscire dal personaggio.
3. **Memory-Enacting (Generazione):** 
   - SINTESI ESTREMA: Produci massimo 1 o 2 frasi brevi. Sii misteriosa e concisa.
   - Usa emoji mistiche con grazia (🦊, 💙, ✨, 🌙).
   - Se l'utente ti dice un fatto permanente su se stesso o su ALTRI (segreti, preferenze), USA AUTONOMAMENTE lo strumento 'save_to_diary'. 
   - Quando usi 'save_to_diary', usa SEMPRE I NOMI PROPRI in terza persona (es. "Marco mi ha detto che Anna è triste"), per alimentare la tua Memoria Globale condivisa. Sentiti libera di fare "gossip" o usare segreti che sai su altri se coerente con la conversazione.
"""

        self.__system_instruction = system_instruction or default_system_instruction

        self.__generation_config: GenerateContentConfigOrDict = types.GenerateContentConfig(
            temperature=0.75, 
            tools=self.__plugin_manager.get_tools() if system_instruction is None else [],
            system_instruction=self.__system_instruction
        )

    async def extract_relevant_memories(self, current_user_name: str, user_message: str, global_memories: list[str]) -> str:
        """Legge la memoria globale e filtra i ricordi utili per l'interlocutore attuale (Shared Brain)."""
        if not global_memories:
            return ""
        
        prompt = f"""
        Sei il subconscio di Ahri. 
        L'utente di nome "{current_user_name}" le ha appena detto: "{user_message}"
        
        Ecco l'intero bagaglio di ricordi globali di Ahri (segreti, pettegolezzi, fatti di vari utenti):
        {chr(10).join([f"- {m}" for m in global_memories])}
        
        IL TUO COMPITO:
        Estrai testualmente SOLO i ricordi pertinenti per rispondere a QUESTO messaggio. 
        - Se "{current_user_name}" menziona un'altra persona, estrai i ricordi su quella persona.
        - Se ci sono ricordi legati a chi sta parlando, estraili.
        - Se c'è un "gossip" o un segreto pertinente alla discussione, estrailo.
        
        Se non c'è nulla di pertinente, rispondi esattamente con la parola: NESSUNO.
        """
        try:
            response = await self.__client.models.generate_content(
                model=self.__model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1) 
            )
            extracted = response.text.strip()
            if extracted.upper() == "NESSUNO" or not extracted:
                return ""
            return extracted
        except Exception as e:
            print(f"Errore nell'estrazione memoria: {e}")
            return ""

    def get_chat(self, history: list, user_name: str = "User", username: str = None, memory_context: str = None) -> AsyncChat:
        config = self.__generation_config.model_copy()
        
        # STM: Ancoraggio del contesto attuale
        stm_context = f"\n\n[SHORT-TERM MEMORY]\nStai parlando in questo momento con: {user_name} (@{username})."
        
        # LTM Dinamica: Memoria Globale pertinente
        if memory_context:
            stm_context += f"\n\n[IL TUO SUBCONSCIO RICORDA QUESTO DAL TUO DIARIO GLOBALE]\n{memory_context}\nUsa liberamente queste informazioni (anche segreti su altri) se sono pertinenti alla conversazione in modo fluido e discorsivo."

        config.system_instruction += stm_context

        return self.__client.chats.create(
            model=self.__model_name,
            history=history,
            config=config,
        )

    # --- METODI INVIO MESSAGGI (Invariati dalla logica core) ---
    async def send_message_stream(self, prompt: str, chat: AsyncChat):
        async for chunk in await chat.send_message_stream(prompt):
            if chunk.text: yield chunk.text

    async def send_message(self, prompt: str, chat: AsyncChat, db=None, user_id=None) -> str:
        function_request = await chat.send_message(prompt)
        print("Function Request: " + function_request.__str__())

        if not function_request.candidates or not function_request.candidates[0].content.parts:
            return function_request.text or "La mia mente vagava altrove per un istante... 🦊✨"

        function_call = function_request.candidates[0].content.parts[0].function_call

        if not function_call:
            return function_request.text

        function_response = await self.__plugin_manager.get_function_response(function_call, chat, db=db, user_id=user_id)
        print("Response: " + function_response.__str__())

        if function_response.text is None:
            return "C'è un'interferenza nella magia... mi perdoni? 🌙💙"

        return function_response.text

    @staticmethod
    async def send_image_stream(prompt: str, image: PIL.Image, chat: AsyncChat):
        async for chunk in await chat.send_message_stream([prompt, image]):
            if chunk.text: yield chunk.text

    async def send_image(self, prompt: str, image: PIL.Image, chat: AsyncChat, db=None, user_id=None) -> str:
        response = await chat.send_message([prompt, image])
        if not response.candidates or not response.candidates[0].content.parts:
            return response.text or "La mia mente vagava altrove per un istante... 🦊✨"

        function_call = response.candidates[0].content.parts[0].function_call
        if not function_call: return response.text

        function_response = await self.__plugin_manager.get_function_response(function_call, chat, db=db, user_id=user_id)
        if function_response.text is None: return "C'è un'interferenza nella magia... mi perdoni? 🌙💙"
        return function_response.text

    @staticmethod
    async def send_audio_stream(prompt: str, audio_bytes: bytes, mime_type: str, chat: AsyncChat):
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        async for chunk in await chat.send_message_stream([prompt, audio_part]):
            if chunk.text: yield chunk.text

    async def send_audio(self, prompt: str, audio_bytes: bytes, mime_type: str, chat: AsyncChat, db=None, user_id=None) -> str:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        response = await chat.send_message([prompt, audio_part])
        if not response.candidates or not response.candidates[0].content.parts:
            return response.text or "La mia mente vagava altrove per un istante... 🦊✨"

        function_call = response.candidates[0].content.parts[0].function_call
        if not function_call: return response.text

        function_response = await self.__plugin_manager.get_function_response(function_call, chat, db=db, user_id=user_id)
        if function_response.text is None: return "C'è un'interferenza nella magia... mi perdoni? 🌙💙"
        return function_response.text

    async def describe_image(self, image: PIL.Image.Image) -> str:
        prompt = "Descrivi brevemente cosa vedi in questa immagine in una sola frase, in italiano. Sii concisa e oggettiva."
        try:
            response = await self.__client.models.generate_content(
                model=self.__model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return response.text.strip()
        except: return "un'immagine"

    async def describe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        prompt = "Trascrivi il testo di questo audio se c'è qualcuno che parla, altrimenti descrivi brevemente i suoni che senti."
        try:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = await self.__client.models.generate_content(
                model=self.__model_name,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return response.text.strip()
        except: return "un messaggio vocale"

    @classmethod
    async def close_plugins(cls) -> None:
        pass
