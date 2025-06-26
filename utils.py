import os
import re
import pandas as pd
import streamlit as st
import unicodedata
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import cosine_similarity

# --- Normalização de strings: remove acentos, lower, underscores ---
def normalize_string(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return s.replace(" ", "_")

# --- Cliente Groq para chat completions ---
client = Groq(api_key=st.secrets["GROQ_API"])

# --- Carrega índices vetoriais com cache ---
@st.cache_resource(show_spinner=False)
def carregar_retriever(path: str):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    store = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    return store.as_retriever()

retriever_faq = carregar_retriever("vectorstore/faq_index")
retriever_pdf = carregar_retriever("vectorstore/legal_index")
retriever_planos = carregar_retriever("vectorstore/planos_index")

# --- Busca exata no FAQ, filtrando primeiro por curso ---
def buscar_faq_exata(pergunta: str):
    curso_usuario = normalize_string(st.session_state.get("curso", ""))
    # busca ampla
    docs = retriever_faq.vectorstore.similarity_search(pergunta, k=20)
    # separa específicos x geral
    esp = [d for d in docs if normalize_string(d.metadata.get("curso")) == curso_usuario]
    candidatos = esp if esp else [d for d in docs if normalize_string(d.metadata.get("curso")) == "geral"]
    if not candidatos:
        return None

    # shortcut para perguntas sobre horas
    if "hora" in pergunta.lower():
        bloco = max(
            (d for d in candidatos if "hora" in d.page_content.lower()),
            key=lambda d: len(d.page_content),
            default=None
        )
        if bloco:
            return bloco

    emb_perg = retriever_faq.vectorstore.embedding_function.embed_query(pergunta)
    melhor, best_score = None, -1.0
    for d in candidatos:
        emb_doc = retriever_faq.vectorstore.embedding_function.embed_query(d.page_content)
        score = cosine_similarity([emb_perg], [emb_doc])[0][0]
        if score > best_score:
            melhor, best_score = d, score

    return melhor if best_score > 0.85 else None

# --- Filtra documentos por curso (mantém 'geral') ---
def filtrar_por_curso(docs, curso_usuario: str):
    norm_u = normalize_string(curso_usuario)
    return [
        d for d in docs
        if normalize_string(d.metadata.get("curso", "")) in (norm_u, "geral")
    ]

# --- Responde ao usuário com RAG + fallback FAQ ---
def responder_usuario(pergunta: str):
    if not (retriever_faq and retriever_pdf and retriever_planos):
        return (
            "⚠️ Meus índices ainda estão carregando. "
            "Envie as pastas `faq_index`, `legal_index` e `planos_index` e clique em 'Rerun'.",
            False
        )

    raw_curso   = st.session_state.get("curso", "")
    curso_title = raw_curso.replace("_", " ").title() if raw_curso else ""
    ctx_user    = f"O usuário é do curso {curso_title}.\n" if curso_title else ""

    # 1) Tenta resposta exata via FAQ
    doc_exato = buscar_faq_exata(pergunta)
    if doc_exato:
        # limpa número e metadado
        texto = doc_exato.page_content
        texto = texto.split("metadado:")[0]                    # remove tudo após "metadado:"
        texto = re.sub(r"^\s*\d+\.\s*", "", texto).strip()     # remove prefixo "N. "
        resp  = f"🤗 Claro! {texto} 😊"
        return resp, True

    # 2) Resto do RAG: busca + filtro por curso
    docs_faq    = filtrar_por_curso(retriever_faq.invoke(pergunta), raw_curso)
    docs_pdf    = retriever_pdf.invoke(pergunta)
    docs_planos = filtrar_por_curso(retriever_planos.invoke(pergunta), raw_curso)

    if not (docs_faq or docs_pdf or docs_planos):
        return (
            "🤔 Não encontrei nada nos meus arquivos. "
            "Anotei sua dúvida e vou repassar para a coordenação!",
            False
        )

    # 3) Concatena conteúdos para contexto
    todos    = docs_faq[:2] + docs_pdf[:2] + docs_planos[:3]
    contexto = "\n\n".join(d.page_content for d in todos)[:15000]

    # 4) Histórico das últimas 6 mensagens
    historico = ""
    if st.session_state.get("chat_history"):
        ult = st.session_state.chat_history[-6:]
        historico = "\n".join(f"{e['role']}: {e['content']}" for e in ult)

    # 5) Monta prompt para Groq
    system = (
        f"{ctx_user}"
    "Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.\n"
    "Responda com simpatia, emoji e objetividade. Baseie-se apenas no contexto fornecido e nos documentos internos.\n"
    "Não consulte fontes externas nem invente informações.\n"
    "Se a informação não estiver no contexto, oriente o usuário a consultar o site oficial: https://www.ifsudestemg.edu.br/barbacena.\n"
    )
    user = f"""
Histórico:
{historico}

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""
    rsp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return rsp.choices[0].message.content.strip(), True

# --- Registra perguntas não respondidas ---
def registrar_pergunta_nao_respondida(pergunta: str):
    os.makedirs("data", exist_ok=True)
    path = "data/nao_respondido.csv"
    if not os.path.exists(path):
        pd.DataFrame(columns=["pergunta"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    if pergunta not in df["pergunta"].values:
        df.loc[len(df)] = pergunta
        df.to_csv(path, index=False)