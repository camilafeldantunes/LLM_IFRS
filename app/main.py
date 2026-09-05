from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from app.vision import classify_image
from app.rag import responder

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

    return {
        "classificacao": classificacao,
        **resultado_rag,
    }