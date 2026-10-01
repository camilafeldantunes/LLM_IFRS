from pathlib import Path

import streamlit as st
from vision import classify_image
from rag import responder
from historico import registrar
import uuid

# Requer streamlit >= 1.43 (st.chat_input com accept_file)

AVATAR = str(Path(__file__).parent / "assets" / "cora_avatar.png")
PERGUNTA_PADRAO = "Como devo descartar este objeto corretamente?"

st.set_page_config(page_title="CORA", page_icon=AVATAR, layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: #F4F7F1; }
    .cora-logo { font-size: 2.6rem; font-weight: 900; letter-spacing: 2px;
                 color: #1F4D36; margin: 0; line-height: 1; }
    .cora-logo span { color: #5BA34F; }
    .cora-sub { color: #1F4D36; opacity: .8; margin: .2rem 0 1rem 0; }
    [data-testid="stChatMessage"] { background: #FFFFFF; border-radius: 18px;
                                    border: 1px solid #E3EADF; }
    </style>
    <p class="cora-logo">C<span>O</span>RA</p>
    <p class="cora-sub">Como descartar? Pergunte ou envie uma foto.</p>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "classificacao_atual" not in st.session_state:
    st.session_state.classificacao_atual = None
if "sessao_id" not in st.session_state:
    st.session_state.sessao_id = uuid.uuid4().hex[:8]

with st.sidebar:
    st.image(AVATAR, width=100)
    if st.button("🗑️ Nova conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.classificacao_atual = None
        st.rerun()

for message in st.session_state.messages:
    avatar = AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("image"):
            st.image(message["image"], width=260)
        st.markdown(message["content"])

entrada = st.chat_input(
    "Pergunte para a CORA...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
)

if entrada:
    texto = (entrada.text or "").strip()
    arquivo = entrada.files[0] if entrada.files else None
    if arquivo and not texto:
        texto = PERGUNTA_PADRAO

    if texto:
        imagem_bytes = arquivo.getvalue() if arquivo else None
        st.session_state.messages.append(
            {"role": "user", "content": texto, "image": imagem_bytes}
        )
        with st.chat_message("user"):
            if imagem_bytes:
                st.image(imagem_bytes, width=260)
            st.markdown(texto)

        with st.chat_message("assistant", avatar=AVATAR):
            with st.spinner("Analisando..."):
                prefixo = ""
                if imagem_bytes:
                    st.session_state.classificacao_atual = classify_image(
                        imagem_bytes, media_type=arquivo.type
                    )
                    c = st.session_state.classificacao_atual
                    prefixo = f"🔎 **{c.get('objeto')}** ({c.get('categoria')})\n\n"

                res = responder(
                    texto, classificacao_visao=st.session_state.classificacao_atual
                )
                registrar(
                    pergunta=texto,
                    resposta=res["resposta"],
                    fontes=res.get("fontes"),
                    classificacao=st.session_state.classificacao_atual,
                    tem_imagem=bool(imagem_bytes),
                    origem="streamlit",
                    sessao_id=st.session_state.sessao_id,
                )

                resposta_final = prefixo + res["resposta"]
                if res.get("fontes"):
                    resposta_final += f"\n\n*Fontes: {', '.join(res['fontes'])}*"

                st.markdown(resposta_final)
                st.session_state.messages.append(
                    {"role": "assistant", "content": resposta_final}
                )