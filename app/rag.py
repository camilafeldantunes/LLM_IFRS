"""
Módulo RAG: recupera trechos relevantes da base de conhecimento e
gera a resposta final combinando (opcionalmente) o resultado da visão.
"""
import chromadb
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    TEXT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)

_client = OpenAI(api_key=OPENAI_API_KEY)
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection(COLLECTION_NAME)

SYSTEM_PROMPT = """
Você é um assistente especialista em resíduos sólidos e descarte inteligente.
Responda de forma clara, prática e curta, sempre indicando:
1) a categoria correta do resíduo,
2) como descartá-lo corretamente (ex: lavar antes, separar componentes),
3) para onde levar (coleta seletiva comum, ecoponto, logística reversa, etc.),
   caso essa informação esteja no contexto fornecido.

Use APENAS as informações do contexto recuperado abaixo. Se o contexto não
cobrir a pergunta, diga isso claramente e dê uma orientação geral segura,
deixando claro que é uma orientação genérica.
"""


def _embed(texto: str) -> list[float]:
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=[texto])
    return response.data[0].embedding


def buscar_contexto(pergunta: str, top_k: int = TOP_K) -> list[dict]:
    """Busca os trechos mais relevantes no vector DB."""
    embedding_pergunta = _embed(pergunta)

    resultados = _collection.query(
        query_embeddings=[embedding_pergunta],
        n_results=top_k,
    )

    trechos = []
    documentos = resultados.get("documents", [[]])[0]
    metadatas = resultados.get("metadatas", [[]])[0]
    for texto, meta in zip(documentos, metadatas):
        trechos.append({"texto": texto, "fonte": meta.get("fonte", "desconhecida")})
    return trechos


def responder(pergunta: str, classificacao_visao: dict | None = None) -> dict:
    """
    Gera a resposta final. Se `classificacao_visao` for passado (vindo do
    módulo vision.py), ele é incorporado ao prompt como contexto adicional.
    """
    trechos = buscar_contexto(pergunta)
    contexto_texto = "\n\n".join(
        f"[Fonte: {t['fonte']}]\n{t['texto']}" for t in trechos
    ) or "Nenhum trecho relevante encontrado na base."

    contexto_visao = ""
    if classificacao_visao:
        contexto_visao = f"""
Resultado da análise de imagem do objeto:
- Objeto identificado: {classificacao_visao.get('objeto')}
- Categoria sugerida: {classificacao_visao.get('categoria')}
- Confiança: {classificacao_visao.get('confianca')}
- Observações: {classificacao_visao.get('observacoes')}
"""

    prompt_usuario = f"""
{contexto_visao}
Pergunta do usuário: {pergunta}

Contexto recuperado da base de conhecimento:
{contexto_texto}
"""

    resposta = _client.chat.completions.create(
        model=TEXT_MODEL,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario},
        ],
    )

    return {
        "resposta": resposta.choices[0].message.content,
        "fontes": list({t["fonte"] for t in trechos}),
    }
