"""
Módulo de reconhecimento visual de resíduos.

Estratégia atual (Opção A do esqueleto): usar um VLM (modelo multimodal
da OpenAI, ex. GPT-4o) diretamente para identificar o objeto e sugerir
a categoria de descarte.

Se depois você quiser mais precisão, plugue aqui um detector dedicado
(ex.: YOLOv8 fine-tunado em TACO/TrashNet) ANTES de chamar o VLM, e passe
o recorte + a classe detectada como contexto extra no prompt abaixo.
"""
import base64
import json
from openai import OpenAI

from app.config import OPENAI_API_KEY, VISION_MODEL, CATEGORIAS_RESIDUOS

client = OpenAI(api_key=OPENAI_API_KEY)

CLASSIFICATION_PROMPT = f"""
Você é um classificador de resíduos sólidos. Olhe a imagem enviada e identifique
o objeto principal descartado.

Categorias possíveis: {", ".join(CATEGORIAS_RESIDUOS)}

Responda APENAS em JSON válido, sem markdown, seguindo este formato exato:
{{
  "objeto": "nome curto do objeto identificado",
  "categoria": "uma das categorias da lista acima",
  "confianca": "alta" | "media" | "baixa",
  "observacoes": "detalhes relevantes, ex: sujidade, se está quebrado, se é composto por múltiplos materiais",
  "materiais_mistos": true | false
}}
"""


def _image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def classify_image(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """
    Recebe os bytes de uma imagem (ou de um frame extraído de vídeo)
    e retorna a classificação estruturada do resíduo.
    """
    image_b64 = _image_to_base64(image_bytes)

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
                            "url": f"data:{media_type};base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "objeto": "desconhecido",
            "categoria": "rejeito (não reciclável)",
            "confianca": "baixa",
            "observacoes": f"Falha ao interpretar resposta do modelo: {raw_text}",
            "materiais_mistos": False,
        }


def classify_video_frames(frames_bytes: list[bytes], media_type: str = "image/jpeg") -> dict:
    """
    Recebe uma lista de frames (já extraídos do vídeo, ex. 1 a cada 2s)
    e retorna a classificação mais frequente/confiável entre eles.

    Extração de frames fica fora deste módulo (ex.: com ffmpeg/opencv),
    para manter este arquivo focado só na parte de IA.
    """
    resultados = [classify_image(f, media_type) for f in frames_bytes]

    # Estratégia simples: pega a classificação de maior confiança.
    ordem_confianca = {"alta": 3, "media": 2, "baixa": 1}
    melhor = max(resultados, key=lambda r: ordem_confianca.get(r.get("confianca", "baixa"), 0))
    melhor["frames_analisados"] = len(frames_bytes)
    return melhor
