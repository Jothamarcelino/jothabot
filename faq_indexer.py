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
    pergunta = str(row[0])
    resposta_html = str(row[1])
    palavras_chave = str(row[2]) if len(row) > 2 else ""
    
    # Campo que será usado para a busca semântica
    texto_para_busca = pergunta + " " + palavras_chave
    
    doc = Document(
        page_content=resposta_html,
        metadata={"input": texto_para_busca}
    )
    docs.append(doc)

# Divide os documentos em pedaços
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria e salva FAISS
db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local("vectorstore/faq_index")

print("✅ Vetorização do FAQ concluída.")