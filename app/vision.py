import base64
import io
import json
from PIL import Image
from openai import OpenAI

from app.config import OPENAI_API_KEY, VISION_MODEL, CATEGORIAS_RESIDUOS

client = OpenAI(api_key=OPENAI_API_KEY)

# Tamanho máximo (lado maior) da imagem enviada ao modelo. Fotos de celular
# costumam vir em 3000-4000px, o que aumenta MUITO o custo em tokens no
# GPT-4o sem melhorar a classificação de um objeto de descarte.
MAX_DIMENSAO = 1024
QUALIDADE_JPEG = 80

CLASSIFICATION_PROMPT = f"""
Você é um classificador especializado em resíduos sólidos. Analise a imagem enviada e identifique
o objeto principal descartado.

Categorias permitidas: {", ".join(CATEGORIAS_RESIDUOS)}

Responda ESTRITAMENTE em formato JSON com as seguintes chaves:
{{
  "objeto": "nome curto do objeto",
  "categoria": "uma das categorias da lista",
  "confianca": "alta | media | baixa",
  "observacoes": "detalhes como sujeira, avarias ou materiais mistos",
  "materiais_mistos": true ou false
}}
"""


def _redimensionar_imagem(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Reduz a imagem para no máximo MAX_DIMENSAO no lado maior e recomprime
    como JPEG, para diminuir o custo em tokens da API de visão. Retorna os
    bytes já processados e o novo media_type.
    """
    try:
        imagem = Image.open(io.BytesIO(image_bytes))
        imagem = imagem.convert("RGB")  # remove canal alpha/CMYK, garante JPEG válido

        largura, altura = imagem.size
        maior_lado = max(largura, altura)
        if maior_lado > MAX_DIMENSAO:
            escala = MAX_DIMENSAO / maior_lado
            nova_largura = int(largura * escala)
            nova_altura = int(altura * escala)
            imagem = imagem.resize((nova_largura, nova_altura), Image.LANCZOS)

        buffer = io.BytesIO()
        imagem.save(buffer, format="JPEG", quality=QUALIDADE_JPEG, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        # Se por algum motivo não der pra processar (formato estranho etc.),
        # manda a imagem original em vez de quebrar a classificação.
        return image_bytes, "image/jpeg"


def _image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def classify_image(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """
    Recebe os bytes de uma imagem e retorna a classificação estruturada.
    """
    image_bytes, media_type = _redimensionar_imagem(image_bytes)
    image_b64 = _image_to_base64(image_bytes)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CLASSIFICATION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                # "low" mantém custo fixo e baixo (~85 tokens);
                                # suficiente para reconhecer o objeto principal
                                # de uma foto já redimensionada.
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        return json.loads(raw_text)

    except Exception as e:
        return {
            "objeto": "desconhecido",
            "categoria": "rejeito (não reciclável)",
            "confianca": "baixa",
            "observacoes": f"Erro na análise de imagem: {str(e)}",
            "materiais_mistos": False,
        }