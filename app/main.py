from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from vision import classify_image
from rag import responder
from historico import registrar

app = FastAPI(title="Assistente de Resíduos Sólidos - Haystack 2.x")


class ChatRequest(BaseModel):
    pergunta: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest):
    """Pergunta pura em texto usando Haystack RAG."""
    resultado = responder(payload.pergunta)
    registrar(
        pergunta=payload.pergunta,
        resposta=resultado["resposta"],
        fontes=resultado.get("fontes"),
        origem="api",
    )
    return resultado


@app.post("/classify-image")
async def classify_image_endpoint(
    imagem: UploadFile = File(...),
    pergunta: str = Form(default="Como devo descartar este objeto corretamente?"),
):
    """
    1) Classifica o objeto via GPT-4o Vision.
    2) Injeta a classificação na Pipeline Haystack para resposta final.
    """
    image_bytes = await imagem.read()
    media_type = imagem.content_type or "image/jpeg"

    classificacao = classify_image(image_bytes, media_type=media_type)
    resultado_rag = responder(pergunta, classificacao_visao=classificacao)
    registrar(
        pergunta=pergunta,
        resposta=resultado_rag["resposta"],
        fontes=resultado_rag.get("fontes"),
        classificacao=classificacao,
        tem_imagem=True,
        origem="api",
    )

    return {
        "classificacao": classificacao,
        **resultado_rag,
    }