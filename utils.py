import os
import pandas as pd
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Tenta carregar vetor com tratamento de erro
def tentar_carregar_retriever(path):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        retriever = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever()
        return retriever
    except Exception as e:
        print(f"⚠️ Não foi possível carregar vetores de: {path} → {e}")
        return None

# Carrega os três repositórios vetoriais (FAQ, Leis, PPCs)
retriever_faq = tentar_carregar_retriever("vectorstore/faq_index")
retriever_pdf = tentar_carregar_retriever("vectorstore/legal_index")
retriever_planos = tentar_carregar_retriever("vectorstore/planos_index")

# Cliente Groq com fallback para secrets
client = Groq(api_key=os.environ.get("GROQ_API", st.secrets["GROQ_API"]))

# Função principal de resposta
def responder_usuario(pergunta):
    if not retriever_faq and not retriever_pdf and not retriever_planos:
        return (
            "⚠️ Os arquivos vetoriais ainda não foram carregados. "
            "Acesse a aba 'Files' e envie as pastas `vectorstore/faq_index/`, `legal_index/` e `planos_index/`. "
            "Depois clique em 'Rerun'.", False
        )

    # 🔍 Prioriza FAQ
    docs_faq = retriever_faq.get_relevant_documents(pergunta)[:1] if retriever_faq else []

    if docs_faq:
        contexto = "\n\n".join([doc.page_content for doc in docs_faq])
    else:
        # Caso FAQ não encontre, busca nas leis e planos
        docs_pdf = retriever_pdf.get_relevant_documents(pergunta)[:1] if retriever_pdf else []
        docs_planos = retriever_planos.get_relevant_documents(pergunta)[:1] if retriever_planos else []

        if not docs_pdf and not docs_planos:
            return ("🤔 Hmm... não encontrei nada sobre isso nos meus arquivos. Mas já registrei sua dúvida! 😉", False)

        contexto = "\n\n".join([doc.page_content for doc in docs_pdf + docs_planos])

    # 🔒 Limita tamanho
    contexto = contexto[:15000]

    # 🧠 Prompt reforçado
    prompt = f"""
Você é o JOTHA, assistente virtual da Coordenação de Estágio do IF Sudeste MG - Campus Barbacena.

Responda com simpatia, clareza e **base apenas no contexto abaixo**. Não invente, não complemente e não improvise.  
Seja útil e direto, mas mantenha um tom acolhedor e divertido.

⚠️ Regras:
- Use exatamente o que estiver no contexto, especialmente se houver HTML.
- Se não encontrar resposta, diga que não encontrou e oriente o usuário a procurar a Coordenação de Estágio.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Você responde em português, com gentileza, precisão e sem inventar informações."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip(), True

# Registra perguntas não respondidas
def registrar_pergunta_nao_respondida(pergunta):
    arquivo = "nao_respondidas.csv"
    if os.path.exists(arquivo):
        df = pd.read_csv(arquivo)
    else:
        df = pd.DataFrame(columns=["pergunta"])

    nova_linha = pd.DataFrame([{"pergunta": pergunta}])
    df = pd.concat([df, nova_linha], ignore_index=True)
    df.to_csv(arquivo, index=False)