import os
import re
import pandas as pd
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from difflib import get_close_matches

# Inicializa cliente Groq
client = Groq(api_key=st.secrets["GROQ_API"])

# Função auxiliar para carregar vetores
@st.cache_resource(show_spinner=False)
def carregar_retriever(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever()

# Carrega os retrievers
retriever_faq = carregar_retriever("vectorstore/faq_index")
retriever_pdf = carregar_retriever("vectorstore/legal_index")
retriever_planos = carregar_retriever("vectorstore/planos_index")

# Armazena informações importantes da sessão
informacoes_chave = {
    "curso": None,
    "nome": None,
    "tipo_estagio": None,
}

# Entrevista inicial
def entrevista_inicial():
    st.markdown("### 👋 Olá, eu sou o JOTHA!")
    st.markdown("Seja bem-vindo ao assistente da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena! 🎓")

    if "nome" not in st.session_state:
        nome = st.text_input("👤 Qual é o seu nome?")
        if nome:
            st.session_state["nome"] = nome
            st.rerun()
        st.stop()

    if "curso" not in st.session_state:
        curso = st.text_input("🎓 Qual o seu curso?")
        if curso:
            st.session_state["curso"] = curso
            st.rerun()
        st.stop()

    if "tipo_estagio" not in st.session_state:
        tipo = st.radio("📄 Qual tipo de atividade você tem dúvidas?", ["obrigatório", "não obrigatório", "horas complementares"])
        if tipo:
            st.session_state["tipo_estagio"] = tipo
            st.rerun()
        st.stop()

    st.success("Pronto, agora é só mandar sua pergunta! 💬")

# Exibe informações memorizadas para o usuário
def exibir_resumo_memoria():
    st.markdown("### ℹ️ Informações que já memorizo nesta sessão:")
    curso = st.session_state.get("curso", "🔍 Não identificado")
    nome = st.session_state.get("nome", "🕵️ Não informado")
    tipo = st.session_state.get("tipo_estagio", "📄 Não especificado")

    st.markdown(f"""
    - 👤 **Nome:** {nome}
    - 🎓 **Curso:** {curso}
    - 📄 **Tipo de Estágio:** {tipo}
    """)

# Extrai e memoriza informações da pergunta

def memorizar_informacoes_chave(pergunta):
    texto = pergunta.lower()

    if "meu nome é" in texto:
        partes = texto.split("meu nome é")
        if len(partes) > 1:
            nome = partes[1].strip().split()[0]
            st.session_state["nome"] = nome

    if "obrigatório" in texto:
        st.session_state["tipo_estagio"] = "obrigatório"
    elif "não obrigatório" in texto or "nao obrigatório" in texto:
        st.session_state["tipo_estagio"] = "não obrigatório"

    documentos = retriever_planos.vectorstore.docstore._dict.values()
    cursos_existentes = list({doc.metadata.get("curso", "") for doc in documentos})
    melhores = get_close_matches(texto, cursos_existentes, n=1, cutoff=0.5)
    if melhores:
        st.session_state["curso"] = melhores[0]

# Função principal de resposta

def responder_usuario(pergunta):
    memorizar_informacoes_chave(pergunta)

    if not retriever_faq and not retriever_pdf and not retriever_planos:
        return (
            "⚠️ Os arquivos vetoriais ainda não foram carregados. "
            "Acesse a aba 'Files' e envie as pastas `vectorstore/faq_index/`, `legal_index/` e `planos_index/`. "
            "Depois clique em 'Rerun'.", False
        )

    # --- 🔍 Etapa 1: Busca no FAQ com verificação de similaridade
    resultados_faq = retriever_faq.vectorstore.similarity_search_with_score(pergunta, k=1)
    if resultados_faq:
        doc, score = resultados_faq[0]
        if score > 0.85:
            return doc.page_content.strip(), True  # 🔁 Retorno imediato da resposta HTML do FAQ

    # --- 🔍 Etapa 2: Busca complementar nos PDFs e Planos
    docs_pdf = retriever_pdf.get_relevant_documents(pergunta)[:1] if retriever_pdf else []
    docs_planos = retriever_planos.get_relevant_documents(pergunta)[:4] if retriever_planos else []

    curso_detectado = st.session_state.get("curso")
    if curso_detectado:
        docs_planos = [doc for doc in docs_planos if curso_detectado.lower() in doc.metadata.get("curso", "").lower()]

    if not docs_pdf and not docs_planos:
        return ("🤔 Hmm... não encontrei nada sobre isso nos meus arquivos. Mas já registrei sua dúvida! 😉", False)

    # --- 🔧 Etapa 3: Monta o prompt com os documentos encontrados
    contexto = "\n\n".join([doc.page_content for doc in docs_pdf + docs_planos])[:15000]

    prompt = f"""
Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.
Seja divertido, responda com simpatia, use emoticons. As respostas devem ser com clareza e base apenas no contexto abaixo.
Sua missão é **responder somente com base no contexto abaixo**, que foi recuperado dos documentos oficiais e da base de perguntas frequentes. **Não invente, não complemente e não improvise, mas seja divertido.**

- Se a resposta estiver no contexto, use exatamente o que estiver escrito lá.
- Se não encontrar a resposta no contexto, diga educadamente que não encontrou e oriente o usuário a entrar em contato com a Coordenação de Estágio.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Você responde em português, com gentileza e precisão."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip(), True


# Função para registrar perguntas não respondidas

def registrar_pergunta_nao_respondida(pergunta):
    if "nao_respondido.csv" not in os.listdir("data"):
        pd.DataFrame(columns=["pergunta"]).to_csv("data/nao_respondido.csv", index=False)

    df = pd.read_csv("data/nao_respondido.csv")
    if pergunta not in df["pergunta"].values:
        df.loc[len(df)] = [pergunta]
        df.to_csv("data/nao_respondido.csv", index=False)