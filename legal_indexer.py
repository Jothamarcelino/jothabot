from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Caminho da pasta com PDFs jurídicos
pasta_pdfs = "data/legal/"

# Lista todos os arquivos PDF da pasta
pdfs = [f for f in os.listdir(pasta_pdfs) if f.endswith(".pdf")]

all_docs = []

for pdf in pdfs:
    caminho_pdf = os.path.join(pasta_pdfs, pdf)
    loader = PyPDFLoader(caminho_pdf)
    docs = loader.load()
    all_docs.extend(docs)

# Quebra os documentos em pedaços menores
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Gera os embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria e salva o índice vetorial FAISS
db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local("vectorstore/legal_index")

print("✅ Vetorização dos PDFs jurídicos concluída.")