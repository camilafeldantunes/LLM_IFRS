"""
Configurações centrais do projeto.
Todas as chaves/segredos vêm de variáveis de ambiente (.env).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenAI (visão, texto e embeddings) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Vector DB ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "residuos_conhecimento")

# --- RAG ---
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Categorias padrão de resíduos (ajuste conforme sua legislação local) ---
CATEGORIAS_RESIDUOS = [
    "papel/papelão",
    "plástico",
    "vidro",
    "metal",
    "orgânico",
    "eletrônico (e-lixo)",
    "pilha/bateria",
    "perigoso/químico",
    "rejeito (não reciclável)",
]
