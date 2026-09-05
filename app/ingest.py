import os
import glob
from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from app.config import (
    OPENAI_API_KEY,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

DOCS_DIR = "data/docs"

def indexar():
    print("Iniciando ingestão com Haystack 2.x...")

    # 1. Instância do banco vetorial Chroma
    document_store = ChromaDocumentStore(
        persist_path=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    # 2. Leitura dos arquivos locais
    arquivos = glob.glob(os.path.join(DOCS_DIR, "**/*.txt"), recursive=True) + \
              glob.glob(os.path.join(DOCS_DIR, "**/*.md"), recursive=True)

    if not arquivos:
        print("Nenhum arquivo .txt ou .md encontrado em data/docs.")
        return

    documentos = []
    for caminho in arquivos:
        with open(caminho, "r", encoding="utf-8") as f:
            documentos.append(
                Document(content=f.read(), meta={"fonte": os.path.basename(caminho)})
            )

    # 3. Construção e conexão do Pipeline
    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200, split_overlap=30))
    indexing_pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=EMBEDDING_MODEL
        )
    )
    indexing_pipeline.add_component(
        "writer",
        # OVERWRITE evita erro de duplicidade quando você roda a ingestão
        # mais de uma vez sobre a mesma coleção.
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    )

    # Conecta as saídas às entradas dos componentes seguintes
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # 4. Execução do Pipeline
    indexing_pipeline.run({"splitter": {"documents": documentos}})
    print("Indexação concluída com sucesso via Haystack!")

if __name__ == "__main__":
    indexar()