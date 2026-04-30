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
                
                # Basato sulla documentazione API di VoxCPM:
                # [0] target_text: string
                # [1] control_instruction: string (vuoto per Ultimate Cloning)
                # [2] reference_audio: FileData
                # [3] ultimate_cloning_mode: boolean (true)
                # [4] transcript: string (testo di Ahri)
                # [5] cfg: number (2.3)
                # [6] reference_enhancement: boolean (false)
                # [7] text_normalization: boolean (false)
                payload = {
                    "data": [
                        clean_target_text,  # [0] Target Text
                        "",                 # [1] Control Instruction
                        file_data,          # [2] Reference Audio
                        True,               # [3] Ultimate Cloning Mode
                        self.ref_text,      # [4] Transcript of Reference Audio
                        2.3,                # [5] CFG
                        False,              # [6] Reference audio enhancement
                        False               # [7] Text normalization
                    ],
                    "fn_index": 0,          # L'endpoint principale /generate è fn_index 0
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
                                        # Il risultato finale deve contenere un oggetto con "orig_name" o un dizionario FileData
                                        # Gli eventi precedenti con "update" vanno saltati
                                        for output_data in event["output"]["data"]:
                                            if isinstance(output_data, dict) and output_data.get("meta", {}).get("_type") == "gradio.FileData":
                                                output_file = output_data["url"]
                                                # 5. Scarichiamo l'audio generato
                                                final_audio_resp = await client.get(output_file)
                                                final_audio_resp.raise_for_status()
                                                return final_audio_resp.content
                                            elif isinstance(output_data, dict) and "url" in output_data:
                                                output_file = output_data["url"]
                                                final_audio_resp = await client.get(output_file)
                                                final_audio_resp.raise_for_status()
                                                return final_audio_resp.content
                                        
                                        print(f"Nessun file audio trovato nell'output finale: {event}")
                                        return None
                                    else:
                                        print(f"Errore generazione TTS: {event}")
                                        return None
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                print(f"Errore durante il TTS: {e}")
                return None
