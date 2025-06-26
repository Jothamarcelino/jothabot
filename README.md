# JOTHABOT 2.0

O **JOTHA** é um assistente virtual desenvolvido para a Coordenação de estágio do IF Sudeste MG – Campus Barbacena que usa técnicas de RAG (Retrieval‐Augmented Generation) para responder dúvidas de alunos com base em:

- **FAQ institucional**  
- **Legislação** (diretrizes e normativas)  
- **PPC (Projetos Pedagógicos de Curso)**  

Tudo orquestrado em uma interface **Streamlit** com:

- **HuggingFaceEmbeddings** (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)  
- **FAISS** (vetorização e busca por similaridade)  
- **API Groq** para geração final de texto  

---

## 🔍 Principais Funcionalidades

1. **Seleção de Curso**  
   - Dropdown que lista todos os cursos (técnicos e superiores).  
   - O usuário escolhe uma vez, e as buscas RAG são filtradas por esse metadado.

2. **Chat Interativo**  
   - Saudação dinâmica pelo horário do dia.  
   - Histórico de conversas mantido na sessão.  
   - Campo de input (`st.chat_input`) com resposta imediata.

3. **Retrieval-Augmented Generation**  
   - **Busca Exata no FAQ** (similaridade + heurística de “horas”).  
   - **Vetorização via FAISS** para:  
     - Documentos PDF de legislação  
     - PPCs  
   - **GROQ API** recebe o contexto concatenado + histórico recente e gera a resposta.

4. **Transparência & Fallbacks**  
   - Rodapé fixo informando que “JOTHA é uma IA e pode conter imprecisões”.  
   - Se não encontrar nos conteúdos internos, orienta a consultar o site oficial:

     > https://www.ifsudestemg.edu.br/barbacena

5. **Painel Administrativo** (sidebar protegida por senha)  
   - Lista de perguntas não respondidas (CSV).  
   - Download desse CSV para análise posterior.

---

## 📂 Estrutura do Repositório

```text
.
├── app.py                   # Front-end Streamlit
├── utils.py                 # Funções de RAG, busca e Groq
├── planos_indexer.py        # Script de indexação dos PPCs
├── faq_indexer.py           # indexador do FAQ
├── legal_indexer.py         # indexador das leis
├── data/                    
|   ├── faq.pdf              # PDFs do FAQ 
│   ├── planos/              # PDFs lei e regulamentos  
|   ├── legal/               # PDFs dos PPCs
│   └── nao_respondido.csv   # Perguntas que não tiveram resposta
├── vectorstore/
│   ├── faq_index/           # Persistência FAISS para FAQ
│   ├── legal_index/         # FAISS de legislação
│   └── planos_index/        # FAISS dos PPCs
├── requirements.txt         # Dependências Python
└── README.md
```

---

## ⚙️ Instalação e Setup

1. **Clone o repositório**  
   ```bash
   git clone https://github.com/seu-usuario/jothabot.git
   cd jothabot
   ```

2. **Crie e ative um virtualenv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Instale as dependências**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as credenciais (Streamlit Secrets)**  
   Em `.streamlit/secrets.toml` defina:
   ```toml
   [GROQ_API]
   key = "seu-groq-api-key"

   [admin]
   acesso = "senha_do_painel"
   ```

5. **Indexe (ou reindexe) seus documentos**  
   - **FAQ**: execute seu script de indexação do FAQ (se houver).  
   - **PPCs**:
     ```bash
     python planos_indexer.py
     ```
   Isso criará/atualizará `vectorstore/planos_index`.

---

## ▶️ Como Rodar

```bash
streamlit run app.py
```

- A página vai abrir em `http://localhost:8501` (ou endereço fornecido pelo Streamlit Cloud).  
- Se usar o **Streamlit Community Cloud**, apenas faça o deploy apontando para este repositório.  

---

## 🛠️ Personalização

- **Adicionar novos cursos**: edite o dicionário `courses` em `app.py`.  
- **Atualizar PPCs ou FAQ**: coloque os PDFs em `data/planos/` (ou a pasta do FAQ) e reexecute os indexers.  
- **Alterar modelo ou parâmetros**: ajuste o bloco `client.chat.completions.create(...)` em `utils.py`.  
- **Mudar estilo/CSS**: personalize o bloco `<style>` no topo de `app.py`.

---

## 🚧 Boas Práticas

- **Não versionar pastas `vectorstore/`** (são grandes) – prefira carregar no deploy CI/CD.  
- Limpe cache no Streamlit Cloud se atingir limites de memória.  
- Mantenha seus PDFs atualizados e com qualidade de texto (OCR quando necessário).

---

## 🫱🏻‍🫲🏼 INPIRAÇÃO
- Este projeto foi inspirado em [IAssistente Sócrates](https://github.com/viniciusrpb/ia_socrates)

