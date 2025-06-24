import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Diretório com os PPCs
DIRETORIO_PLANOS = "data/planos"
SAIDA_VECTORSTORE = "vectorstore/planos_index"

# Função para detectar automaticamente o nome do curso
def extrair_nome_do_curso(texto: str) -> str:
    texto = texto.replace("\n", " ").replace("  ", " ").upper()

    padroes = [
        r"T[ÉE]CNICO EM ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"BACHARELADO EM ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"LICENCIATURA EM ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"GRADUAÇÃO EM ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"TECNOLOGIA EM ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"CURSO DE ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"CURSO SUPERIOR EM ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
        r"PROJETO PEDAG[ÓO]GICO DO CURSO DE ((?:[A-ZÀ-Ÿ]{2,}(?: [A-ZÀ-Ÿ]{2,}){0,5}))",
    ]

    for padrao in padroes:
        match = re.search(padrao, texto)
        if match:
            nome = match.group(1).strip()

            # Remoção de sufixos irrelevantes
            lixo = [
                "MODALIDADE", "CAMPUS", "PROJETO", "INSTITUTO",
                "COORDENADOR", "PROFESSOR", "REITOR",
                "DEPARTAMENTO", "NÚCLEO", "UNIDADE"
            ]
            for palavra in lixo:
                if palavra in nome:
                    nome = nome.split(palavra)[0].strip()

            return nome.lower().replace(" ", "_")

    # Fallback: busca por termos conhecidos
    termos_famosos = ["AGRONOMIA", "TURISMO", "QUÍMICA", "COMPUTAÇÃO", "ENFERMAGEM", "ALIMENTOS"]
    for termo in termos_famosos:
        if termo in texto:
            return termo.lower().replace(" ", "_")

    return "desconhecido"

# Inicializa embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Coleta documentos vetorizados
todos_chunks = []

for nome_arquivo in os.listdir(DIRETORIO_PLANOS):
    caminho = os.path.join(DIRETORIO_PLANOS, nome_arquivo)
    if not nome_arquivo.endswith(".pdf"):
        continue

    print(f"📄 Lendo {nome_arquivo}...")

    loader = PyPDFLoader(caminho)
    docs = loader.load()

    # Usa as 2 primeiras páginas para detectar o nome do curso
    primeiras_paginas = " ".join([doc.page_content for doc in docs[:2]])
    nome_curso = extrair_nome_do_curso(primeiras_paginas)
    print(f"🔍 Curso identificado: {nome_curso}")

    # Adiciona metadado do curso
    for doc in docs:
        doc.metadata["curso"] = nome_curso

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    todos_chunks.extend(chunks)

# Cria a base vetorial
print("⚙️ Gerando base vetorial dos planos...")
db = FAISS.from_documents(todos_chunks, embedding=embeddings)
db.save_local(SAIDA_VECTORSTORE)
print("✅ Vetorização dos Planos de Curso concluída com metadados automáticos!")