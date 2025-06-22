from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Caminho da pasta com os planos de curso (PDFs)
pasta_planos = "data/planos/"

# Lista os arquivos PDF da pasta
pdfs = [f for f in os.listdir(pasta_planos) if f.endswith(".pdf")]

all_docs = []

for pdf in pdfs:
    caminho_pdf = os.path.join(pasta_planos, pdf)
    loader = PyPDFLoader(caminho_pdf)
    docs = loader.load()
    all_docs.extend(docs)

# Divide os documentos em pedaços menores
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Gera os embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria e salva o índice vetorial FAISS
db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local("vectorstore/planos_index")

print("✅ Vetorização dos Planos de Curso concluída.")