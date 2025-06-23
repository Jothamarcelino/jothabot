import streamlit as st
from utils import responder_usuario, registrar_pergunta_nao_respondida, exibir_resumo_memoria, entrevista_inicial
import pandas as pd

st.set_page_config(page_title="JOTHA 2.0", layout="wide")
st.title("🤖 JOTHA 2.0 - Assistente da Coordenação de Estágio")

st.sidebar.image("https://bit.ly/jotha-aberto", width=100)
st.sidebar.markdown("### 📊 Painel de Gerenciamento")
senha = st.sidebar.text_input("🔐 Digite a senha do painel:", type="password")
acesso_autorizado = senha == st.secrets["admin"]["acesso"]

# Executa a entrevista inicial com o usuário
entrevista_inicial()

# Exibe resumo da sessão
exibir_resumo_memoria()

st.markdown("""
## 🤖 Faça sua pergunta ao JOTHA:
Digite abaixo sua dúvida relacionada ao estágio. O JOTHA responderá com base na legislação, FAQ e PPCs dos cursos.
""")

# Entrada do usuário
pergunta = st.chat_input("Digite sua pergunta sobre estágio, leis ou PPCs...")


if pergunta:
    with st.chat_message("Usuário"):
        st.markdown(pergunta)

    with st.spinner("JOTHA pensando 🤔..."):
        resposta, encontrado = responder_usuario(pergunta)
        with st.chat_message("JOTHA"):
            st.markdown(resposta, unsafe_allow_html=True)

        if not encontrado:
            registrar_pergunta_nao_respondida(pergunta)

# Painel de gerenciamento (se autorizado)
if acesso_autorizado:
    st.markdown("---")
    st.subheader("📥 Perguntas não respondidas registradas:")
    try:
        df = pd.read_csv("nao_respondidas.csv")
        st.dataframe(df)
        st.download_button("📁 Baixar CSV", df.to_csv(index=False), file_name="nao_respondidas.csv")
    except FileNotFoundError:
        st.info("Nenhuma pergunta registrada ainda.")
