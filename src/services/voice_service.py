# FILE: src/services/voice_service.py
import os
import uuid
import json
import httpx
import asyncio
import re

class VoiceService:
    def __init__(self):
        self.base_url = "https://openbmb-voxcpm-demo.hf.space"
        self.hf_token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
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

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                # 1. Scarichiamo l'audio di riferimento di Ahri
                audio_resp = await client.get(self.ref_audio_url)
                audio_resp.raise_for_status()
                audio_bytes = audio_resp.content

                # 2. Carichiamo il file sull'endpoint /upload di Gradio
                upload_url = f"{self.base_url}/gradio_api/upload"
                files = {"files": ("ahri_ref.ogg", audio_bytes, "audio/ogg")}
                upload_resp = await client.post(upload_url, files=files, headers=self.headers)
                upload_resp.raise_for_status()
                file_path = upload_resp.json()[0] # Ritorna il path nel server di HF

                # Creiamo l'oggetto file richiesto da Gradio
                file_data = {
                    "path": file_path,
                    "meta": {"_type": "gradio.FileData"},
                    "orig_name": "ahri_ref.ogg"
                }

                # 3. Ci uniamo alla coda di generazione
                session_hash = uuid.uuid4().hex
                
                # NOTA: fn_index e la struttura di "data" dipendono dallo Space.
                # Solitamente l'ordine è [Audio di riferimento, Testo di riferimento, Testo da generare]
                # Se l'API restituisce errore, controlla l'endpoint /config dello space per l'ordine esatto.
                payload = {
                    "data": [
                        file_data,          # Reference Audio
                        self.ref_text,      # Reference Text
                        clean_target_text   # Target Text
                    ],
                    "fn_index": 0,          # Cambia questo indice se lo Space ha aggiornato la sua UI
                    "session_hash": session_hash
                }

                join_url = f"{self.base_url}/gradio_api/queue/join"
                join_resp = await client.post(join_url, json=payload, headers=self.headers)
                join_resp.raise_for_status()

                # 4. Ascoltiamo gli eventi (Server-Sent Events) per ottenere il risultato
                stream_url = f"{self.base_url}/gradio_api/queue/data?session_hash={session_hash}"
                
                async with client.stream("GET", stream_url, headers=self.headers) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                event = json.loads(data_str)
                                if event.get("msg") == "process_completed":
                                    if event.get("success"):
                                        # Il risultato è nel campo output.data[0] (dipende dall'output dello space)
                                        output_file = event["output"]["data"][0]["url"] 
                                        
                                        # 5. Scarichiamo l'audio generato
                                        final_audio_resp = await client.get(output_file)
                                        final_audio_resp.raise_for_status()
                                        return final_audio_resp.content
                                    else:
                                        print(f"Errore generazione TTS: {event}")
                                        return None
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                print(f"Errore durante il TTS: {e}")
                return None
