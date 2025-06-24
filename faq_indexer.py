import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import os
import shutil

# Caminho do CSV
csv_path = "data/faq.csv"

# Carrega o CSV com nomes das colunas
df = pd.read_csv(csv_path)

docs = []

for _, row in df.iterrows():
    pergunta = str(row["PERGUNTA PRINCIPAL"]).strip()
    resposta_html = str(row["RESPOSTA"]).strip()
    palavras_chave = str(row["Palavra-chave/Tema"]).strip() if "Palavra-chave/Tema" in row else ""
    
    # Texto que será vetorizado (pergunta + palavras-chave)
    texto_para_busca = pergunta + " " + palavras_chave
    
    doc = Document(
        page_content=resposta_html,
        metadata={"input": texto_para_busca}
    )
    docs.append(doc)

# Divide os documentos em pedaços pequenos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Gera os embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Apaga vetor antigo
if os.path.exists("vectorstore/faq_index"):
    shutil.rmtree("vectorstore/faq_index")

# Cria e salva o vetor FAISS
db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local("vectorstore/faq_index")

print("✅ Vetorização do FAQ concluída.")