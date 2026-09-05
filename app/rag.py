from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from app.config import (
    OPENAI_API_KEY,
    TEXT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    CATEGORIAS_RESIDUOS,
)

document_store = ChromaDocumentStore(
    persist_path=CHROMA_DIR,
    collection_name=COLLECTION_NAME
)

template_prompt = """
Você é um especialista em gestão de resíduos sólidos e descarte sustentável. 
Sua missão é instruir o usuário sobre como descartar itens de forma correta, prática e segura.

CATEGORIAS OFICIAIS PERMITIDAS:
{{ categorias }}

---
DIRETRIZES DE RESPOSTA:
1. ESTRUTURA OBRIGATÓRIA DA RESPOSTA (se houver informação no contexto):
   - 🏷️ **Categoria:** [Uma das categorias oficiais da lista acima]
   - 🧼 **Preparo/Cuidados:** [Instruções de limpeza, segurança ou separação antes do descarte]
   - 📍 **Destinação Correta:** [Lixeira colorida correspondente, Ecoponto, Logística Reversa, etc.]

2. REGRAS DE ANÁLISE E GROUNDING (RAG):
   - Baseie suas instruções EXCLUSIVAMENTE nos dados do "Contexto Recuperado".
   - Você DEVE fazer deduções diretas e lógicas de material (exemplo: se a pergunta for sobre "caixa de pizza" e o contexto trouxer "papelão sujo/engordurado", aplique a regra descrita para papelão engordurado).
   - Se o objeto identificado na imagem (`contexto_visao`) for diferente da pergunta em texto, priorize o objeto da imagem, mas mencione a dúvida ao usuário.
   - SE O CONTEXTO NÃO COBRIR O ITEM: Diga exatamente: "Não encontrei instruções específicas para este item na minha base de conhecimento." Em seguida, forneça apenas uma recomendação genérica de segurança, deixando claro que não veio da base oficial.

---
{% if contexto_visao %}
[ANÁLISE DE IMAGEM RECEBIDA]
{{ contexto_visao }}

{% endif %}
[CONTEXTO RECUPERADO DA BASE DE CONHECIMENTO]
{% if documents %}
{% for doc in documents %}
---
(Fonte: {{ doc.meta.fonte }})
{{ doc.content }}
{% endfor %}
{% else %}
(Nenhum documento relevante encontrado na base de conhecimento.)
{% endif %}
---

Pergunta do Usuário: {{ pergunta }}

Resposta:
"""


def criar_rag_pipeline():
    rag_pipeline = Pipeline()

    rag_pipeline.add_component("text_embedder", OpenAITextEmbedder(
        api_key=Secret.from_token(OPENAI_API_KEY),
        model=EMBEDDING_MODEL
    ))
    rag_pipeline.add_component("retriever", ChromaEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K
    ))
    rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template_prompt))
    # Usamos OpenAIGenerator (não o Chat) porque o PromptBuilder entrega uma
    # string de prompt pronta, e não uma lista de ChatMessage.
    rag_pipeline.add_component("llm", OpenAIChatGenerator(
        api_key=Secret.from_token(OPENAI_API_KEY),
        model=TEXT_MODEL
    ))

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    # Em versões recentes do haystack-ai, o input do gerador se chama
    # "messages" (aceita list[ChatMessage] OU str), não mais "prompt".
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    return rag_pipeline


rag_pipeline = criar_rag_pipeline()


def responder(pergunta: str, classificacao_visao: dict | None = None) -> dict:
    contexto_visao = ""
    if classificacao_visao:
        contexto_visao = (
            f"Objeto: {classificacao_visao.get('objeto')}\n"
            f"Categoria: {classificacao_visao.get('categoria')}\n"
            f"Confiança: {classificacao_visao.get('confianca')}\n"
            f"Observações: {classificacao_visao.get('observacoes')}"
        )

    resultado = rag_pipeline.run(
        {
            "text_embedder": {"text": pergunta},
            "prompt_builder": {
                "pergunta": pergunta,
                "contexto_visao": contexto_visao,
                "categorias": ", ".join(CATEGORIAS_RESIDUOS)
            }
        },
        # O retriever não é "folha" do pipeline (sua saída alimenta o
        # prompt_builder), então por padrão o Haystack não devolveria os
        # documentos recuperados. Isso precisa ser pedido explicitamente.
        include_outputs_from={"retriever"}
    )

    resposta_bruta = resultado["llm"]["replies"][0]
    # Em algumas versões do haystack-ai o gerador devolve objetos ChatMessage
    # em vez de string pura, mesmo no OpenAIGenerator "não-chat".
    resposta_texto = resposta_bruta.text if hasattr(resposta_bruta, "text") else str(resposta_bruta)
    documentos_recuperados = resultado["retriever"]["documents"]
    fontes = list({doc.meta.get("fonte", "desconhecida") for doc in documentos_recuperados})

    return {
        "resposta": resposta_texto,
        "fontes": fontes
    }