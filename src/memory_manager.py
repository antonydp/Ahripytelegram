import os
from mem0 import Memory

# mem0 usa Gemini sia per capire i fatti che per gli embedding

config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemma-4-26b-a4b-it",
            "api_key": os.getenv("GEMINI_API_KEY"),
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-2-preview",
            "api_key": os.getenv("GEMINI_API_KEY"),
            "embedding_dims": 768,
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            # mem0 vuole una connessione sync
            "connection_string": os.getenv("SQLALCHEMY_DATABASE_URI", "").replace("postgresql+asyncpg", "postgresql").replace("sqlite+aiosqlite", "sqlite"),
            "collection_name": "ahri_memories",
            "embedding_model_dims": 768,
        }
    }
}

# singleton
ahri_memory = Memory.from_config(config)
