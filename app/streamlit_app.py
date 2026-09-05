import streamlit as st
from app.vision import classify_image
from app.rag import responder



st.set_page_config(
    page_title="EcoGuia - Descarte Inteligente",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ EcoGuia - Assistente de Resíduos")
st.caption("Identifique objetos e descubra a forma correta de descarte.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "classificacao_cache" not in st.session_state:
    st.session_state.classificacao_cache = None
if "classificacao_arquivo_id" not in st.session_state:
    st.session_state.classificacao_arquivo_id = None

# Sidebar para Upload de Imagem
with st.sidebar:
    st.header("📸 Análise por Imagem")
    uploaded_file = st.file_uploader("Envie uma foto do objeto para identificar:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Imagem enviada", use_container_width=True)
        # file_id muda se o usuário trocar de arquivo, mesmo mantendo o nome
        arquivo_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if arquivo_id != st.session_state.classificacao_arquivo_id:
            # Nova imagem: limpa o cache para forçar reclassificação só desta vez
            st.session_state.classificacao_cache = None
            st.session_state.classificacao_arquivo_id = arquivo_id
    else:
        st.session_state.classificacao_cache = None
        st.session_state.classificacao_arquivo_id = None

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
if prompt := st.chat_input("Ex: Como descarto lâmpadas fluorescentes?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            classificacao = None
            if uploaded_file:
                if st.session_state.classificacao_cache is not None:
                    # Já classificamos essa imagem antes nesta sessão — reaproveita.
                    classificacao = st.session_state.classificacao_cache
                else:
                    bytes_data = uploaded_file.getvalue()
                    classificacao = classify_image(bytes_data, media_type=uploaded_file.type)
                    st.session_state.classificacao_cache = classificacao
                st.info(f"**Objeto Identificado:** {classificacao.get('objeto')} ({classificacao.get('categoria')})")

            res = responder(prompt, classificacao_visao=classificacao)
            
            resposta_final = res["resposta"]
            if res.get("fontes"):
                resposta_final += f"\n\n---\n**Fontes consultadas:** {', '.join(res['fontes'])}"
            
            st.markdown(resposta_final)
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})