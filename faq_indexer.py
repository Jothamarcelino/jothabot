import os
import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Configuração do logging para acompanhar as etapas
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Caminho do PDF do FAQ
pdf_path = "data/faq.pdf"
logging.info(f"Carregando PDF do FAQ: {pdf_path}")

# Carrega o PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()
logging.info(f"Total de páginas carregadas: {len(pages)}")

# Junta todas as páginas em um único texto
conteudo_total = "\n".join([p.page_content for p in pages])
logging.info("Conteúdo total do PDF reunido.")

# Divide o conteúdo por seções numeradas (ex: "11. Curso Técnico em Alimentos")
blocos = re.split(r"(?=\n\d{1,3}\.\s)", conteudo_total)
logging.info(f"Número de blocos identificados: {len(blocos)}")

# Expressão regular para capturar nome de curso (caso não haja metadado explícito)
regex_curso = re.compile(
    r"(CURSO(S)? (T[ÉE]CNICO|SUPERIOR|DE|EM|LICENCIATURA|BACHARELADO)[^\n]*)",
    re.IGNORECASE
)

# Função para extrair o metadado do curso do texto
def extrair_nome_do_curso(texto: str) -> str:
    # Primeiro, tenta capturar explicitamente o metadado definido no fim do bloco
    m_meta = re.search(r"metadado:\s*([^\n]+)", texto, re.IGNORECASE)
    if m_meta:
        valor = m_meta.group(1).strip().lower()
        logging.info(f"Metadado explícito detectado: {valor}")
        return valor
    # Se não houver metadado explícito, usa o regex para extrair o nome do curso
    match = regex_curso.search(texto)
    if match:
        nome = match.group(0)
        logging.info(f"Curso detectado bruto: {nome}")
        nome = re.sub(r"CURSOS? ", "", nome, flags=re.IGNORECASE)
        nome = nome.strip().lower().replace(" ", "_")
        logging.info(f"Curso normalizado: {nome}")
        return nome
    logging.info("Nenhum curso específico detectado; atribuindo 'geral'")
    return "geral"

# Converte os blocos em objetos Document com metadado de curso
docs = []
for idx, bloco in enumerate(blocos, start=1):
    texto = bloco.strip()
    if len(texto) < 50:
        logging.debug(f"Bloco {idx} ignorado por ser muito curto.")
        continue
    curso_detectado = extrair_nome_do_curso(texto)
    metadados = {"fonte": "faq", "curso": curso_detectado}
    logging.info(f"Bloco {idx}: Metadados: {metadados}")
    doc = Document(page_content=texto, metadata=metadados)
    docs.append(doc)

logging.info(f"📄 Total de blocos vetorizados: {len(docs)}")

# Gera os embeddings
logging.info("Gerando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Cria a base vetorial FAISS
logging.info("Criando o índice vetorial FAISS...")
db = FAISS.from_documents(docs, embedding=embeddings)

# Salva localmente a base vetorial
os.makedirs("vectorstore/faq_index", exist_ok=True)
db.save_local("vectorstore/faq_index")

logging.info("✅ FAQ vetorizado com metadados por curso!")