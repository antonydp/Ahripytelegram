import os
import logging
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

try:
    from mem0 import Memory
    MEM0_ENABLED = True
except ImportError as e:
    logging.error(f"mem0 import failed: {e}")
    MEM0_ENABLED = False
    Memory = None

def _sanitize_pg_connection_string(raw_url: str) -> str:
    """
    Converte postgresql+asyncpg://...?ssl=true in formato psycopg2:
    postgresql://...?sslmode=require
    """
    if not raw_url:
        return raw_url

    # 1. Togli il driver asyncpg
    url = raw_url.replace("postgresql+asyncpg://", "postgresql://")

    # 2. Parsa e sistema i parametri query
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    # psycopg2 vuole sslmode, non ssl
    if "ssl" in query:
        ssl_val = query.pop("ssl")[0]
        if ssl_val.lower() in ("true", "1", "require"):
            query["sslmode"] = ["require"]

    # Ricostruisci la URL
    new_query = urlencode(query, doseq=True)
    sanitized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    return sanitized

ahri_memory = None

if MEM0_ENABLED:
    try:
        raw_db_url = os.getenv("SQLALCHEMY_DATABASE_URI", "")
        pg_dsn = _sanitize_pg_connection_string(raw_db_url)

        config = {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "temperature": 0.1,
                    "custom_fact_extraction_prompt": """
                    Agisci come il diario segreto di Ahri. Analizza la conversazione e salva SOLO informazioni cruciali, fatti concreti o preferenze importanti.

                    REGOLE RIGIDE:
                    1. Salva SOLO fatti reali: segreti, accordi, prezzi, password, date, preferenze o eventi significativi.
                    2. Ignora TUTTO il resto: chiacchiere, saluti, complimenti, poesie, frasi d'amore, metafore o roleplay.
                    3. Scrivi una sola frase per ogni fatto, iniziando SEMPRE con il nome del soggetto interessato (es: "@username ha detto che...").
                    4. Non usare mai la parola "utente".
                    5. Se non ci sono informazioni importanti da salvare, non produrre alcun output (ritorna un array vuoto di fatti).

                    Rispondi rigorosamente in formato JSON:
                    {"facts": ["frase 1", "frase 2"]}
                    """
                }
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-embedding-2-preview",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "embedding_dims": 768,
                }
            },
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "connection_string": pg_dsn,
                    "collection_name": "ahri_memories",
                    "embedding_model_dims": 768,
                }
            }
        }
        ahri_memory = Memory.from_config(config)
        logging.info("mem0 inizializzato correttamente")
    except Exception as e:
        logging.error(f"mem0 init failed: {e}. Continuo senza memoria.")
        ahri_memory = None
