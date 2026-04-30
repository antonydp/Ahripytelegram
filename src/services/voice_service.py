# FILE: src/services/voice_service.py
import os
import re
import asyncio
from gradio_client import Client, handle_file

class VoiceService:
    def __init__(self):
        self.space_id = "openbmb/VoxCPM-Demo"
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Dati di Ahri
        self.ref_audio_url = "https://lolsound.com/sounds/Ahri/it_IT/base/Ahri_Kill_MissFortune_2236818.ogg"
        self.ref_text = "non c'è debolezza nell'andare avanti, Sara, ci vuole forza"

    def clean_text_for_tts(self, text: str) -> str:
        """Pulisce il testo da emoji e caratteri speciali che potrebbero confondere il TTS."""
        # Rimuove le emoji più comuni che Ahri usa (🦊, ✨, 🌙, 💙, ecc.)
        clean = re.sub(r'[^\w\s,.!?\'"èéàòùì-]', '', text)
        return clean.strip()

    async def generate_voice(self, target_text: str) -> bytes | None:
        clean_target_text = self.clean_text_for_tts(target_text)
        if not clean_target_text:
            return None

        try:
            # Eseguiamo la chiamata client in un thread separato perché gradio_client è sincrono
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_generate, clean_target_text)
            
            if result and os.path.exists(result):
                with open(result, "rb") as f:
                    audio_content = f.read()
                # Rimuoviamo il file temporaneo creato da gradio_client
                try:
                    os.remove(result)
                except:
                    pass
                return audio_content
            
            return None

        except Exception as e:
            print(f"Errore durante il TTS (Gradio Client): {e}")
            return None

    def _sync_generate(self, target_text: str) -> str:
        client = Client(self.space_id)
        result = client.predict(
            text_input=target_text,
            control_instruction="",
            reference_wav_path_input=handle_file(self.ref_audio_url),
            use_prompt_text=True,
            prompt_text_input=self.ref_text,
            cfg_value_input=2.3,
            do_normalize=False,
            denoise=False,
            api_name="/generate"
        )
        return result
