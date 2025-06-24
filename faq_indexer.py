import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import os

# Caminho do CSV
csv_path = "data/faq.csv"

# Carrega o CSV com perguntas (coluna A), respostas (coluna B) e palavras-chave (coluna C)
df = pd.read_csv(csv_path)

docs = []

for _, row in df.iterrows():
    pergunta = str(row[0]).strip()
    resposta_html = str(row[1]).strip()
    palavras_chave = str(row[2]).strip() if len(row) > 2 else ""

    # Texto para vetorização = pergunta + palavras-chave + resposta (mas só usamos a resposta na exibição final)
    texto_para_vetor = f"{pergunta}. Palavras-chave: {palavras_chave}. Resposta: {resposta_html}"

    doc = Document(
        page_content=resposta_html,  # Mostra apenas a resposta
        metadata={"input": texto_para_vetor}  # Usa o conteúdo estendido para a busca vetorial
    )
    docs.append(doc)

# Divide os documentos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Gera embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria e salva FAISS
db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local("vectorstore/faq_index")

print("✅ Vetorização do FAQ concluída.")