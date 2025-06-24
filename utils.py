import os
import pandas as pd
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import cosine_similarity

# Inicializa cliente Groq
client = Groq(api_key=st.secrets["GROQ_API"])

# Carrega os retrievers
@st.cache_resource(show_spinner=False)
def carregar_retriever(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever()

retriever_faq = carregar_retriever("vectorstore/faq_index")
retriever_pdf = carregar_retriever("vectorstore/legal_index")
retriever_planos = carregar_retriever("vectorstore/planos_index")

# Busca exata no FAQ usando similaridade por metadado
def buscar_faq_exata(pergunta):
    docs = retriever_faq.vectorstore.similarity_search(pergunta, k=5)
    if not docs:
        return None

    pergunta_embedding = retriever_faq.vectorstore.embedding_function.embed_query(pergunta)
    melhor_doc = None
    melhor_score = -1

    for doc in docs:
        texto_referencia = doc.metadata.get("input", "")
        doc_embedding = retriever_faq.vectorstore.embedding_function.embed_query(texto_referencia)
        score = cosine_similarity([pergunta_embedding], [doc_embedding])[0][0]
        if score > melhor_score:
            melhor_doc = doc
            melhor_score = score

    if melhor_score > 0.90:
        return melhor_doc
    return None

# Função principal de resposta
def responder_usuario(pergunta):
    if not retriever_faq and not retriever_pdf and not retriever_planos:
        return (
            "⚠️ Os arquivos vetoriais ainda não foram carregados. "
            "Acesse a aba 'Files' e envie as pastas `vectorstore/faq_index/`, `legal_index/` e `planos_index/`. "
            "Depois clique em 'Rerun'.", False
        )

    # Etapa 1: tentativa de resposta exata via FAQ
    doc_exato = buscar_faq_exata(pergunta)
    if doc_exato:
        return doc_exato.page_content.strip(), True

    # Etapa 2: busca complementar nas leis e planos
    docs_pdf = retriever_pdf.get_relevant_documents(pergunta)[:2]
    docs_planos = retriever_planos.get_relevant_documents(pergunta)[:3]

    if not docs_pdf and not docs_planos:
        return ("🤔 Hmm... não encontrei nada sobre isso nos meus arquivos. Mas já registrei sua dúvida! 😉", False)

    contexto = "\n\n".join([doc.page_content for doc in docs_pdf + docs_planos])[:15000]

    prompt = f"""
Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.
Responda com simpatia, use emoticons, e seja direto. Sua resposta deve se basear apenas no contexto abaixo.
⚠️ **Não invente informações. Não improvise. Seja claro, preciso e divertido.**

Se a resposta estiver no contexto, use exatamente o que estiver escrito lá.
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
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip(), True

# Registro de perguntas não respondidas
def registrar_pergunta_nao_respondida(pergunta):
    if "nao_respondido.csv" not in os.listdir("data"):
        pd.DataFrame(columns=["pergunta"]).to_csv("data/nao_respondido.csv", index=False)

    df = pd.read_csv("data/nao_respondido.csv")
    if pergunta not in df["pergunta"].values:
        df.loc[len(df)] = [pergunta]
        df.to_csv("data/nao_respondido.csv", index=False)