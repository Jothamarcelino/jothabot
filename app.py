import time
import random
import datetime
import streamlit as st
from utils import responder_usuario, registrar_pergunta_nao_respondida
import pandas as pd

# --- Cursos e Metadados ---
courses = {
    "Curso Técnico em Alimentos": "alimentos",
    "Curso Técnico em Agroindústria (Integrado)": "agroindústria_integrado_ao_ensino_médio_barbacena",
    "Curso Técnico em Agropecuária (Integrado)": "agropecuária_integrad",
    "Curso Técnico em Hospedagem (Integrado)": "hospedagem_integrado_ao_ensino_médio_barbacena",
    "Curso Técnico em Química (Integrado)": "química_integrad",
    "Curso Técnico em Enfermagem": "enfermagem",
    "Curso Técnico em Informática": "informática",
    "Curso Técnico em Meio Ambiente": "meio_ambiente",
    "Curso Técnico em Segurança do Trabalho": "segurança_do_trabalho",
    "Curso Superior em Administração": "administração",
    "Curso Superior em Agronomia": "agronomia",
    "Curso de Ciências Biológicas (Licenciatura)": "ciências_biológicas",
    "Curso de Química (Licenciatura)": "química",
    "Curso de Educação Física (Licenciatura)": "educação_física",
    "Curso em Gestão Ambiental": "gestão_ambiental",
    "Curso em Gestão de Turismo": "gestao_de_turismo",
    "Curso Superior em Nutrição": "nutrição",
    "Curso Superior em Alimentos (Tecnólogo)": "tecnologia_em_alimentos"
}

def get_welcome_message():
    h = datetime.datetime.now().hour
    if h < 12:
        return random.choice([
            "☀️ Bom dia, explorador do estágio! Pronto para novas descobertas? Eu sou o JOTHA, seu guia de campo!",
            "🌞 Olá, mestre das manhãs! JOTHA por aqui para te ajudar com tudo sobre seu estágio.",
            "😄 Seja bem-vindo ao universo do estágio! Eu sou o JOTHA e as aventuras matinais começam agora!"
        ])
    if h < 18:
        return random.choice([
            "🌤️ Boa tarde, futuro profissional de sucesso! Eu sou o JOTHA, pronto pra te ajudar no que for preciso.",
            "📚 Olá! JOTHA na área! Preparado pra mais um capítulo da sua jornada de estágio?",
            "💼 Boa tarde! Vamos colocar os conhecimentos em prática juntos? O JOTHA te acompanha nessa missão!"
        ])
    return random.choice([
        "🌙 Boa noite, guerreiro do conhecimento! Eu sou o JOTHA, e ainda dá tempo de brilhar no estágio!",
        "🦉 Noite produtiva por aqui? Eu sou o JOTHA, sempre acordado para te ajudar com o estágio!",
        "💡 A luz do conhecimento nunca se apaga... e eu também não! Boa noite, sou o JOTHA!"
    ])

# --- Configuração & CSS ---
st.set_page_config(page_title="JOTHA 2.0", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  body { background-color: #f9f9f9; }
  [data-testid="stChatMessage"][data-testid-chat-message-type="user"] {
    background-color:#ffebcc!important; text-align:right; border-radius:8px!important; padding:8px!important;
  }
  [data-testid="stChatMessage"][data-testid-chat-message-type="assistant"] {
    background-color:#ffffff!important; text-align:left; border-radius:8px!important; padding:8px!important;
  }
</style>
""", unsafe_allow_html=True)
# --- Sidebar de Gerenciamento ---
st.sidebar.image("https://bit.ly/jotha-aberto", width=100)
st.sidebar.markdown("### 📊 Painel de Gerenciamento")
senha = st.sidebar.text_input("🔐 Senha do painel:", type="password")
acesso_autorizado = senha == st.secrets["admin"]["acesso"]

# --- Session State Defaults ---
st.session_state.setdefault("welcome_shown", False)
st.session_state.setdefault("curso", None)
st.session_state.setdefault("curso_exibido", None)
st.session_state.setdefault("chat_history", [])

# --- 1) Saudação + digitação + pergunta de curso (apenas uma vez) ---
if not st.session_state.welcome_shown:
    with st.spinner("JOTHA está digitando…"):
        time.sleep(1)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": get_welcome_message()
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "Para começarmos, qual o seu curso? 🎓"
    })
    st.session_state.welcome_shown = True

# --- 2) Se ainda não escolheu curso: renderiza histórico + select + confirm e para ---
if st.session_state.curso is None:
    for msg in st.session_state.chat_history:
        quem = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(quem):
            st.markdown(msg["content"], unsafe_allow_html=True)

    escolha = st.selectbox("Selecione seu curso", list(courses.keys()))
    if st.button("Confirmar"):
        st.session_state.curso = courses[escolha]
        st.session_state.curso_exibido = escolha
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Ótimo! Você escolheu **{escolha}**. Vamos lá! 🚀"
        })
        # segue a execução para o próximo bloco
    else:
        st.stop()

# --- 3) Curso definido: receber pergunta e registrar no histórico ANTES do loop ---
pergunta = st.chat_input("")
if pergunta:
    st.session_state.chat_history.append({"role": "user", "content": pergunta})
    resposta, encontrado = responder_usuario(pergunta)
    st.session_state.chat_history.append({"role": "assistant", "content": resposta})
    if not encontrado:
        registrar_pergunta_nao_respondida(pergunta)
    # o próprio chat_input dispara o rerun, então não precisamos de st.experimental_rerun()


# --- 4) Exibe todo histórico (1 único loop) ---
for msg in st.session_state.chat_history:
    quem = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(quem):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --- 5) Mudar curso + Sidebar de não-respondidas ---
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(f"**Curso selecionado:** {st.session_state.curso_exibido}")
with col2:
    if st.button("🔄️ Mudar curso"):
        for k in ("welcome_shown", "curso", "curso_exibido", "chat_history"):
            st.session_state.pop(k, None)
        st.stop()


if acesso_autorizado:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Perguntas não respondidas")
    try:
        df = pd.read_csv("data/nao_respondido.csv")
        st.sidebar.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("📁 Baixar nao_respondido.csv", csv, "nao_respondido.csv", "text/csv")
    except FileNotFoundError:
        st.sidebar.info("Nenhuma pergunta registrada.")

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f9f9f9;
    text-align: center;
    font-size: 0.8em;
    color: #666;
    padding: 10px 0;
    border-top: 1px solid #ddd;
    z-index: 100;
}
</style>
<div class="footer">
⚠️ O JOTHA é um assistente IA. As respostas podem conter imprecisões.
</div>
""", unsafe_allow_html=True)
