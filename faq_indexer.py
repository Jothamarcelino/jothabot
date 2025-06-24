import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Caminho do PDF do FAQ
pdf_path = "data/faq.pdf"

# Carrega o PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Junta todas as páginas em um único texto
conteudo_total = "\n".join([p.page_content for p in pages])

# Divide o texto por seções numeradas (1. Título...)
# Garante que não perca o número durante a divisão
blocos = re.split(r"(?=\n\d{1,3}\.\s)", conteudo_total)

# Converte em objetos Document
docs = []
for bloco in blocos:
    texto = bloco.strip()
    if len(texto) < 50:
        continue
    doc = Document(page_content=texto, metadata={"fonte": "faq"})
    docs.append(doc)

print(f"📄 Total de documentos extraídos do FAQ: {len(docs)}")

# Gera os embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria base vetorial FAISS
db = FAISS.from_documents(docs, embedding=embeddings)

# Salva a base vetorial
os.makedirs("vectorstore/faq_index", exist_ok=True)
db.save_local("vectorstore/faq_index")

print("✅ FAQ vetorizado com sucesso a partir do PDF!")
