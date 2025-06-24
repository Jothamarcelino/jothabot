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

# Divide o conteúdo por seções numeradas (ex: "11. Curso Técnico em Alimentos")
blocos = re.split(r"(?=\n\d{1,3}\.\s)", conteudo_total)

# Expressão regular para capturar nome de curso nas seções
regex_curso = re.compile(
    r"(CURSO(S)? (T[ÉE]CNICO|SUPERIOR|DE|EM|LICENCIATURA|BACHARELADO)[^\n]*)", re.IGNORECASE
)

# Função para extrair o nome do curso do texto
def extrair_nome_do_curso(texto: str) -> str:
    match = regex_curso.search(texto)
    if match:
        nome = match.group(0)
        nome = re.sub(r"CURSOS? ", "", nome, flags=re.IGNORECASE)
        nome = nome.strip().lower().replace(" ", "_")
        return nome
    return "geral"

# Converte os blocos em objetos Document com metadado de curso
docs = []
for bloco in blocos:
    texto = bloco.strip()
    if len(texto) < 50:
        continue
    curso_detectado = extrair_nome_do_curso(texto)
    metadados = {"fonte": "faq", "curso": curso_detectado}
    doc = Document(page_content=texto, metadata=metadados)
    docs.append(doc)

print(f"📄 Total de blocos vetorizados: {len(docs)}")

# Gera os embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria a base vetorial FAISS
db = FAISS.from_documents(docs, embedding=embeddings)

# Salva localmente a base vetorial
os.makedirs("vectorstore/faq_index", exist_ok=True)
db.save_local("vectorstore/faq_index")

print("✅ FAQ vetorizado com metadados por curso!")