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

# --- Cliente Groq ---
client = Groq(api_key=st.secrets["GROQ_API"])

# --- Carrega vectorstore (usado para montar retrievers dinâmicos) ---
@st.cache_resource(show_spinner=False)
def carregar_vectorstore(path: str):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)

vectorstore_faq    = carregar_vectorstore("vectorstore/faq_index")
vectorstore_pdf    = carregar_vectorstore("vectorstore/legal_index")
vectorstore_planos = carregar_vectorstore("vectorstore/planos_index")

# --- Busca exata no FAQ, filtrando primeiro por curso ---
def buscar_faq_exata(pergunta: str):
    curso_usuario = normalize_string(st.session_state.get("curso", ""))
    docs = vectorstore_faq.similarity_search(pergunta, k=20)
    esp = [d for d in docs if normalize_string(d.metadata.get("curso")) == curso_usuario]
    candidatos = esp if esp else [d for d in docs if normalize_string(d.metadata.get("curso")) == "geral"]
    if not candidatos:
        return None

    if "hora" in pergunta.lower():
        bloco = max(
            (d for d in candidatos if "hora" in d.page_content.lower()),
            key=lambda d: len(d.page_content),
            default=None
        )
        if bloco:
            return bloco

    emb_perg = vectorstore_faq.embedding_function.embed_query(pergunta)
    melhor, best_score = None, -1.0
    for d in candidatos:
        emb_doc = vectorstore_faq.embedding_function.embed_query(d.page_content)
        score = cosine_similarity([emb_perg], [emb_doc])[0][0]
        if score > best_score:
            melhor, best_score = d, score

    return melhor if best_score > 0.85 else None

# --- Resposta final: monta contexto e envia para modelo ---
def responder_usuario(pergunta: str):
    raw_curso   = st.session_state.get("curso", "")
    curso_title = raw_curso.replace("_", " ").title() if raw_curso else ""
    ctx_user    = f"O usuário é do curso {curso_title}.\n" if curso_title else ""

    if not (vectorstore_faq and vectorstore_pdf and vectorstore_planos):
        return (
            "⚠️ Meus índices ainda estão carregando. "
            "Envie as pastas `faq_index`, `legal_index` e `planos_index` e clique em 'Rerun'.",
            False
        )

    # 1) FAQ exato
    doc_exato = buscar_faq_exata(pergunta)
    if doc_exato:
        texto = doc_exato.page_content
        texto = texto.split("metadado:")[0]
        texto = re.sub(r"^\s*\d+\.\s*", "", texto).strip()
        return f"🤗 Claro! {texto} 😊", True

    # 2) RAG com filtro direto por curso
    retr_faq = vectorstore_faq.as_retriever(search_kwargs={"k": 4, "filter": {"curso": raw_curso}})
    retr_planos = vectorstore_planos.as_retriever(search_kwargs={"k": 4, "filter": {"curso": raw_curso}})
    retr_pdf = vectorstore_pdf.as_retriever(search_kwargs={"k": 4})  # sem filtro

    docs_faq = retr_faq.invoke(pergunta)
    docs_pdf = retr_pdf.invoke(pergunta)
    docs_planos = retr_planos.invoke(pergunta)

    if not (docs_faq or docs_pdf or docs_planos):
        return (
            "🤔 Não encontrei nada nos meus arquivos. "
            "Anotei sua dúvida e vou repassar para a coordenação!",
            False
        )

    todos = docs_faq[:2] + docs_pdf[:2] + docs_planos[:3]
    contexto = "\n\n".join(d.page_content for d in todos)[:15000]

    historico = ""
    if st.session_state.get("chat_history"):
        ult = st.session_state.chat_history[-6:]
        historico = "\n".join(f"{e['role']}: {e['content']}" for e in ult)

    system = (
        f"{ctx_user}"
        "Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.\n"
        "Responda com simpatia, emoji e objetividade. Baseie-se apenas no contexto.\n"
        "Não invente informações.\n"
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