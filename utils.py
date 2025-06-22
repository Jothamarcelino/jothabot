import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os
import streamlit as st

# Carrega vetores com segurança
def carregar_retriever(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever()

retriever_faq = carregar_retriever("vectorstore/faq_index")
retriever_pdf = carregar_retriever("vectorstore/legal_index")
retriever_planos = carregar_retriever("vectorstore/planos_index")

# Cliente da Groq
client = Groq(api_key=os.environ.get("GROQ_API", st.secrets["GROQ_API"]))

# Função principal para responder ao usuário
def responder_usuario(pergunta):
    # Busca nos três vetores
    docs_faq = retriever_faq.invoke(pergunta)
    docs_pdf = retriever_pdf.invoke(pergunta)
    docs_planos = retriever_planos.invoke(pergunta)

    # Se nenhum contexto for encontrado
    if not docs_faq and not docs_pdf and not docs_planos:
        return ("🤔 Hmm... não encontrei nada sobre isso nos meus arquivos. Mas já registrei sua dúvida! 😉", False)

    # Junta os conteúdos encontrados
    contexto = "\n\n".join([doc.page_content for doc in docs_faq + docs_pdf + docs_planos])

    prompt = f"""
Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.
Responda com simpatia, clareza e base apenas no contexto abaixo. Nunca invente nada.

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
    arquivo = "nao_respondidas.csv"
    if os.path.exists(arquivo):
        df = pd.read_csv(arquivo)
    else:
        df = pd.DataFrame(columns=["pergunta"])

    nova_linha = pd.DataFrame([{"pergunta": pergunta}])
    df = pd.concat([df, nova_linha], ignore_index=True)
    df.to_csv(arquivo, index=False)