import os
import pandas as pd
import streamlit as st
import unicodedata
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import cosine_similarity

# Função para normalizar strings (remove acentos, converte para minúsculas e substitui espaços por underline)
def normalize_string(s: str) -> str:
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.replace(" ", "_")

# Inicializa o cliente Groq
client = Groq(api_key=st.secrets["GROQ_API"])

# Função de carregamento de índice vetorial com cache
@st.cache_resource(show_spinner=False)
def carregar_retriever(path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever()

# Carrega os retrievers
retriever_faq = carregar_retriever("vectorstore/faq_index")
retriever_pdf = carregar_retriever("vectorstore/legal_index")
retriever_planos = carregar_retriever("vectorstore/planos_index")

# Busca exata no FAQ com similaridade por metadado e filtro por curso
def buscar_faq_exata(pergunta):
    docs = retriever_faq.vectorstore.similarity_search(pergunta, k=10)
    if not docs:
        return None

    curso_usuario = normalize_string(st.session_state.get("curso", ""))
    pergunta_embedding = retriever_faq.vectorstore.embedding_function.embed_query(pergunta)

    melhor_doc = None
    melhor_score = -1

    for doc in docs:
        curso_doc = normalize_string(doc.metadata.get("curso", "geral"))
        # Considera válido se o documento for "geral" ou se houver correspondência parcial
        if curso_doc != "geral" and (curso_usuario not in curso_doc and curso_doc not in curso_usuario):
            continue

        # Usa o metadado "input" se existir; caso contrário, pega os primeiros 200 caracteres do conteúdo
        texto_referencia = doc.metadata.get("input", "") or doc.page_content[:200]
        doc_embedding = retriever_faq.vectorstore.embedding_function.embed_query(texto_referencia)
        score = cosine_similarity([pergunta_embedding], [doc_embedding])[0][0]

        if score > melhor_score:
            melhor_doc = doc
            melhor_score = score

    if melhor_score > 0.85:
        return melhor_doc
    return None

# Função para filtrar documentos por curso, permitindo "geral" como fallback
def filtrar_por_curso(docs, curso_usuario):
    filtrados = []
    norm_usuario = normalize_string(curso_usuario)
    for doc in docs:
        curso_doc = normalize_string(doc.metadata.get("curso", "geral"))
        if curso_doc == "geral" or norm_usuario in curso_doc or curso_doc in norm_usuario:
            filtrados.append(doc)
    return filtrados

# Função principal de resposta do JOTHA
def responder_usuario(pergunta):
    if not retriever_faq and not retriever_pdf and not retriever_planos:
        return (
            "⚠️ Os arquivos vetoriais ainda não foram carregados. "
            "Acesse a aba 'Files' e envie as pastas `faq_index`, `legal_index` e `planos_index`, depois clique em 'Rerun'.",
            False,
        )

    curso_usuario = st.session_state.get("curso", "").strip()
    contexto_usuario = (
        f"O usuário é do curso {curso_usuario.replace('_', ' ').title()}.\n"
        if curso_usuario
        else ""
    )

    # Tenta responder via FAQ com busca exata
    doc_exato = buscar_faq_exata(pergunta)
    if doc_exato:
        return doc_exato.page_content.strip(), True

    # Recupera documentos dos três retrievers e filtra pelo curso
    docs_faq = filtrar_por_curso(retriever_faq.get_relevant_documents(pergunta), curso_usuario)
    docs_pdf = retriever_pdf.get_relevant_documents(pergunta)
    docs_planos = filtrar_por_curso(retriever_planos.get_relevant_documents(pergunta), curso_usuario)

    # Se nenhum documento for encontrado, retorna mensagem de dúvida
    if not docs_faq and not docs_pdf and not docs_planos:
        return (
            "🤔 Hmm... não encontrei nada sobre isso nos meus arquivos. Mas já registrei sua dúvida! 😉",
            False,
        )

    # Concatena conteúdos dos documentos para formar o contexto, limitando o tamanho
    contexto = "\n\n".join(
        [doc.page_content for doc in (docs_faq[:2] + docs_pdf[:2] + docs_planos[:3])]
    )[:15000]

    historico = ""
    if "chat_history" in st.session_state:
        # Limita ao histórico das últimas 6 mensagens
        historico = "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in st.session_state.chat_history[-6:]]
        )

    prompt = f"""
{contexto_usuario}
Histórico da conversa:
{historico}

Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.
Responda com simpatia, use emoticons e seja direto. Sua resposta deve se basear apenas no contexto abaixo.
⚠️ **Não invente informações. Não improvise. Seja claro, preciso e divertido.**

Se a resposta estiver no contexto, use exatamente o que estiver escrito.
Se não encontrar a resposta, diga que não encontrou e oriente o usuário a entrar em contato com a Coordenação de Estágio.

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
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip(), True

# Registro de perguntas não respondidas
def registrar_pergunta_nao_respondida(pergunta):
    os.makedirs("data", exist_ok=True)
    filepath = "data/nao_respondido.csv"

    if not os.path.exists(filepath):
        pd.DataFrame(columns=["pergunta"]).to_csv(filepath, index=False)

    df = pd.read_csv(filepath)
    if pergunta not in df["pergunta"].values:
        df.loc[len(df)] = [pergunta]
        df.to_csv(filepath, index=False)