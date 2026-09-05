"""
API principal do assistente de resíduos sólidos.

Endpoints:
- POST /chat            -> pergunta em texto, resposta via RAG
- POST /classify-image  -> foto do objeto, resposta via visão + RAG
- GET  /health          -> healthcheck

Rodar localmente:
    uvicorn app.main:app --reload
    http://localhost:8000/docs#/default/health_health_get
"""
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from app.vision import classify_image
from app.rag import responder

app = FastAPI(title="Assistente de Resíduos Sólidos")


class ChatRequest(BaseModel):
    pergunta: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest):
    """Pergunta puramente textual (ex: 'como descarto óleo de cozinha?')."""
    resultado = responder(payload.pergunta)
    return resultado


@app.post("/classify-image")
async def classify_image_endpoint(
    imagem: UploadFile = File(...),
    pergunta: str = Form(default="Como devo descartar este objeto corretamente?"),
):
    """
    Recebe uma foto do objeto + pergunta opcional.
    1) Classifica o objeto via VLM.
    2) Usa a classificação como contexto para o RAG gerar a orientação final.
    """
    image_bytes = await imagem.read()
    media_type = imagem.content_type or "image/jpeg"

    classificacao = classify_image(image_bytes, media_type=media_type)
    resultado_rag = responder(pergunta, classificacao_visao=classificacao)

    return {
        "classificacao": classificacao,
        **resultado_rag,
    }


# --- Placeholder para vídeo ---
# Para vídeo, extraia frames antes de chamar a API (ex.: com ffmpeg/opencv no
# frontend ou num worker separado) e reaproveite /classify-image por frame,
# ou implemente um endpoint /classify-video que receba múltiplos arquivos
# e chame app.vision.classify_video_frames().
