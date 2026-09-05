"""
Script de ingestão da base de conhecimento (RAG).

Coloque seus documentos (legislação, manuais de coleta seletiva,
guias municipais, PDFs convertidos em .txt/.md, etc.) em data/docs/
e rode:

    python -m app.ingest

Isso quebra os textos em chunks, gera embeddings via OpenAI e indexa
no ChromaDB.
"""
import os
import glob
import uuid

import chromadb
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

DOCS_DIR = "data/docs"
client = OpenAI(api_key=OPENAI_API_KEY)


def carregar_textos() -> list[dict]:
    """Lê todos os .txt/.md de data/docs e retorna [{'texto':..., 'fonte':...}]"""
    arquivos = glob.glob(os.path.join(DOCS_DIR, "**/*.txt"), recursive=True) + \
        glob.glob(os.path.join(DOCS_DIR, "**/*.md"), recursive=True)

    documentos = []
    for caminho in arquivos:
        with open(caminho, "r", encoding="utf-8") as f:
            documentos.append({"texto": f.read(), "fonte": os.path.basename(caminho)})
    return documentos


def dividir_em_chunks(texto: str, tamanho: int = CHUNK_SIZE, sobreposicao: int = CHUNK_OVERLAP) -> list[str]:
    """Divisão simples por caracteres com sobreposição. Troque por um splitter
    mais robusto se preferir."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunks.append(texto[inicio:fim])
        inicio += tamanho - sobreposicao
    return [c.strip() for c in chunks if c.strip()]


def gerar_embeddings(textos: list[str]) -> list[list[float]]:
    """Chama a API de embeddings da OpenAI em lote."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=textos)
    return [item.embedding for item in response.data]


def indexar():
    print("Carregando documentos de", DOCS_DIR)
    documentos = carregar_textos()
    if not documentos:
        print("Nenhum documento encontrado em data/docs. Adicione .txt/.md e rode novamente.")
        return

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    total_chunks = 0
    for doc in documentos:
        chunks = dividir_em_chunks(doc["texto"])
        if not chunks:
            continue

        embeddings = gerar_embeddings(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"fonte": doc["fonte"]} for _ in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  - {doc['fonte']}: {len(chunks)} chunks indexados")

    print(f"Concluído. {total_chunks} chunks indexados em '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    indexar()
