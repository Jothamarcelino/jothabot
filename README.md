# 🤖 JOTHA 2.0 – Assistente Jurídico com IA (RAG + Streamlit)

O **JOTHA 2.0** é um chatbot inteligente desenvolvido para auxiliar a Coordenação de Estágio do IF Sudeste MG. Ele utiliza:

- ✅ FAQ institucional com respostas formatadas em HTML
- 📄 Documentos jurídicos (leis, resoluções, planos de curso)
- 🧠 Vetorização semântica com FAISS + Hugging Face
- 🔗 Recuperação aumentada por geração (RAG)
- 🧬 LLMs via API da Groq (como o LLaMA 3)
- 🖥️ Interface amigável em Streamlit

---

## 🚀 Como rodar localmente

### 1. Instale as dependências:

Com Poetry:

```bash
poetry install
Ou com pip:

bash
Copiar
Editar
pip install -r requirements.txt
2. Configure os segredos:
Crie o arquivo .streamlit/secrets.toml com sua chave da Groq e a senha do painel:

toml
Copiar
Editar
GROQ_API = "sua-chave-aqui"

[admin]
acesso = "jotha123"
3. Prepare os dados:
Coloque seu faq.csv em data/

Coloque os PDFs em data/legal/

Adicione planos de curso em data/planos/

Rode os scripts:

bash
Copiar
Editar
python faq_indexer.py
python legal_indexer.py
4. Inicie o app:
bash
Copiar
Editar
streamlit run app.py
🔐 Painel de gerenciamento:
Após digitar a senha, você poderá:

📋 Visualizar perguntas não respondidas

📥 Fazer download do arquivo CSV com sugestões

🛠️ Manter e atualizar o JOTHA com base nas dúvidas reais dos usuários

☁️ Como fazer deploy no Streamlit Cloud:
Faça push do repositório para o GitHub

Acesse https://streamlit.io/cloud

Conecte seu repositório GitHub

Adicione os segredos (GROQ_API, etc.)

Pronto! O JOTHA estará no ar e pronto para ajudar!

Desenvolvido com ❤️ no IF Sudeste MG – Campus Barbacena.