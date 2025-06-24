import streamlit as st
from utils import responder_usuario, registrar_pergunta_nao_respondida
import pandas as pd

# Configuração da página
st.set_page_config(page_title="JOTHA 2.0", layout="wide")
st.title("🤖 JOTHA 2.0 - Assistente da Coordenação de Estágio")

# Sidebar de gerenciamento
st.sidebar.image("https://bit.ly/jotha-aberto", width=100)
st.sidebar.markdown("### 📊 Painel de Gerenciamento")
senha = st.sidebar.text_input("🔐 Digite a senha do painel:", type="password")
acesso_autorizado = senha == st.secrets["admin"]["acesso"]

# Inicializa o estado da sessão para armazenar o curso e o histórico da conversa
if "curso" not in st.session_state:
    st.session_state.curso = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Solicita ao usuário o nome do curso se ainda não informado
if not st.session_state.curso:
    curso_input = st.text_input("Informe o seu curso (ex: Técnico em Química):")
    if curso_input:
        st.session_state.curso = curso_input.strip().lower()
        st.success(f"Curso definido: {st.session_state.curso}")

st.markdown("""
## 🤖 Faça sua pergunta ao JOTHA:
Digite abaixo sua dúvida relacionada ao estágio, leis ou PPCs.
""")

# Entrada de pergunta do usuário
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    # Armazena pergunta no histórico
    st.session_state.chat_history.append({"role": "Usuário", "content": pergunta})
    with st.chat_message("Usuário"):
        st.markdown(pergunta)

    with st.spinner("JOTHA pensando 🤔..."):
        resposta, encontrado = responder_usuario(pergunta)
        
        # Armazena resposta no histórico
        st.session_state.chat_history.append({"role": "JOTHA", "content": resposta})
        with st.chat_message("JOTHA"):
            st.markdown(resposta, unsafe_allow_html=True)
        
        if not encontrado:
            registrar_pergunta_nao_respondida(pergunta)

# Exibe o histórico da conversa para referência
if st.session_state.chat_history:
    st.markdown("---")
    st.write("### Histórico da Conversa")
    for entry in st.session_state.chat_history:
        if entry["role"] == "Usuário":
            st.markdown(f"**Usuário:** {entry['content']}")
        else:
            st.markdown(f"**JOTHA:** {entry['content']}")

# Painel de gerenciamento (apenas se o acesso for autorizado)
if acesso_autorizado:
    st.markdown("---")
    st.subheader("📥 Perguntas não respondidas registradas:")
    try:
        df = pd.read_csv("data/nao_respondido.csv")
        st.dataframe(df)
        st.download_button("📁 Baixar CSV", df.to_csv(index=False), file_name="nao_respondido.csv")
    except FileNotFoundError:
        st.info("Nenhuma pergunta registrada ainda.")