import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "residuos_conhecimento")

TOP_K = int(os.getenv("TOP_K", "4"))

CATEGORIAS_RESIDUOS = [
    "papel/papelão", "plástico", "vidro", "metal",
    "orgânico", "eletrônico (e-lixo)", "pilha/bateria",
    "perigoso/químico", "rejeito (não reciclável)"
]