# 🌱 CORA — Classificação e Orientação para Resíduos Ambientais.

CORA é a assistente virtual do IFRS que ajuda a classificar e destinar corretamente os resíduos sólidos. Com apoio de Inteligência Artificial, ela responde dúvidas por texto, reconhece objetos por foto e orienta o descarte correto com base em uma base de conhecimento própria.

> Inteligência hoje. Um amanhã mais sustentável.

---

## O que o projeto faz

- **Tira dúvidas por texto:** o usuário pergunta como descartar um item (ex.: "Como descarto lâmpadas fluorescentes?").
- **Reconhece objetos por imagem:** o usuário anexa uma foto no chat e a CORA identifica o objeto e sua categoria.
- **Orienta o descarte:** a resposta segue sempre a mesma estrutura: categoria, preparo/cuidados e destinação correta.
- **Responde só com base nos documentos:** a CORA consulta a base de conhecimento (`data/docs`) e cita as fontes usadas. Se o item não estiver coberto, ela avisa em vez de inventar.
- **Registra as conversas** em um banco local, para análise posterior.

### Categorias de resíduos

papel/papelão · plástico · vidro · metal · orgânico · eletrônico (e-lixo) · pilha/bateria · perigoso/químico · rejeito (não reciclável)

---

## Como funciona

```
Texto da pergunta ─┐
                   ├─► Embedding ─► Busca no Chroma ─► Prompt + contexto ─► GPT-4o ─► Resposta + fontes
Foto (opcional) ───┘                                          ▲
   │                                                          │
   └─► GPT-4o Vision ─► objeto, categoria, confiança ─────────┘
```

1. **Visão (`vision.py`):** se houver foto, ela é redimensionada (máx. 1024 px) para reduzir custo e enviada ao modelo de visão, que devolve um JSON com objeto, categoria, confiança e observações.
2. **Recuperação (`rag.py`):** a pergunta vira um embedding e o Chroma devolve os `TOP_K` trechos mais parecidos da base de conhecimento.
3. **Geração (`rag.py`):** um pipeline Haystack monta o prompt (instruções + categorias + análise da imagem + trechos recuperados) e o modelo de texto escreve a resposta.
4. **Registro (`historico.py`):** cada interação é salva em SQLite.

**Tecnologias:** Python, Haystack 2.x, ChromaDB, OpenAI (GPT-4o e `text-embedding-3-small`), Streamlit, FastAPI.

---

## Estrutura do projeto

```
.
├── .streamlit/
│   └── config.toml              # tema verde da CORA
├── app/
│   ├── assets/
│   │   └── cora_avatar.png      # avatar da CORA
│   ├── config.py                # variáveis de configuração e categorias
│   ├── ingest.py                # indexa os documentos no Chroma
│   ├── rag.py                   # pipeline RAG (Haystack)
│   ├── vision.py                # classificação de imagem
│   ├── streamlit_app.py         # interface de chat
│   ├── main.py                  # API FastAPI
│   ├── historico.py             # registro das conversas (SQLite)
│   └── exportar_historico.py    # exporta o histórico para CSV
├── data/
│   ├── docs/                    # base de conhecimento (.md e .txt)
│   ├── chroma/                  # banco vetorial (gerado pelo ingest)
│   └── historico.db             # conversas (gerado automaticamente)
├── .env                         # chave da API (não vai para o Git)
├── requirements.txt
└── README.md
```

---

## Como rodar

### Requisitos

- Python **3.10 ou superior**
- Streamlit **1.43 ou superior** (necessário para anexar imagem dentro do chat)
- Uma chave de API da OpenAI

### 1. Clonar e criar o ambiente virtual

```bash
git clone <url-do-repositorio>
cd <pasta-do-projeto>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

Se preferir instalar manualmente:

```bash
pip install -U streamlit haystack-ai chroma-haystack openai pillow python-dotenv fastapi uvicorn python-multipart
```

### 3. Configurar o `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua-chave-aqui
```

Variáveis opcionais (os valores abaixo são os padrões):

| Variável          | Padrão                   | Para que serve                                |
| ----------------- | ------------------------ | --------------------------------------------- |
| `VISION_MODEL`    | `gpt-4o`                 | Modelo que analisa as fotos                   |
| `TEXT_MODEL`      | `gpt-4o`                 | Modelo que escreve as respostas               |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings da busca                 |
| `CHROMA_DIR`      | `data/chroma`            | Onde o banco vetorial é salvo                 |
| `COLLECTION_NAME` | `residuos_conhecimento`  | Nome da coleção no Chroma                     |
| `TOP_K`           | `4`                      | Quantos trechos recuperar a cada pergunta     |

### 4. Adicionar a base de conhecimento

Coloque arquivos `.md` ou `.txt` em `data/docs/`. Um assunto por arquivo funciona bem (ex.: `pilhas-baterias.md`, `eletronicos.md`). A CORA só responde com base no que estiver aqui.

### 5. Indexar os documentos

```bash
python app/ingest.py
```

Rode de novo sempre que **editar ou adicionar** documentos. Se **remover ou renomear** um arquivo, apague a pasta `data/chroma` antes de reindexar, porque o ingest sobrescreve mas não apaga trechos antigos.

### 6. Abrir o app

```bash
streamlit run app/streamlit_app.py
```

Acesse `http://localhost:8501`. Para enviar uma foto, use o clipe 📎 na caixa de mensagem.

> Rode sempre os comandos **a partir da raiz do projeto**. Os caminhos `data/docs` e `data/chroma` são relativos a ela.

---

## API (opcional)

Para usar a CORA como serviço, a partir da raiz do projeto:

```bash
uvicorn main:app --app-dir app --reload
```

A documentação interativa fica em `http://localhost:8000/docs`.

| Método | Rota              | O que faz                                                             |
| ------ | ----------------- | --------------------------------------------------------------------- |
| GET    | `/health`         | Verifica se a API está no ar                                          |
| POST   | `/chat`           | Recebe `{"pergunta": "..."}` e devolve a resposta e as fontes          |
| POST   | `/classify-image` | Recebe uma `imagem` (e `pergunta` opcional), classifica e responde     |

Exemplo:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Como descarto pilhas?"}'
```

---

## Histórico de conversas

Cada interação (pelo app ou pela API) é gravada em `data/historico.db` com: data e hora, origem, id da sessão, pergunta, se havia imagem, objeto e categoria identificados, confiança, resposta e fontes.

Para analisar em planilha:

```bash
python app/exportar_historico.py
```

Isso gera `data/historico.csv`, que abre no Excel ou Google Planilhas. Dicas de análise: perguntas respondidas sem fonte e casos de confiança "baixa" mostram onde a base de conhecimento precisa crescer.

**Privacidade:** as fotos **não** são salvas, apenas o resultado da classificação. As perguntas e respostas ficam registradas, então vale avisar os usuários caso o app seja aberto ao público.

---

## Problemas comuns

| Sintoma                                            | Causa provável / solução                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `File does not exist: streamlit_app.py`            | Use o caminho completo: `streamlit run app/streamlit_app.py`                                      |
| Erro com `accept_file`                             | Streamlit desatualizado: `pip install -U streamlit`                                              |
| "Nenhum arquivo .txt ou .md encontrado"            | A pasta `data/docs` não existe ou está vazia                                                     |
| Erro 401 da OpenAI                                 | Chave ausente ou incorreta no `.env`                                                             |
| Respostas sem fontes / "não encontrei instruções"  | Falta rodar o `ingest.py`, ou o assunto não está na base                                          |
| `PipelineConnectError` ao abrir o app              | O gerador em `rag.py` precisa ser o `OpenAIGenerator` ligado em `llm.prompt`                      |
| `ModuleNotFoundError` ao rodar a API               | Use `--app-dir app` no comando do uvicorn, a partir da raiz                                      |

---

## Licença e créditos

Projeto desenvolvido no contexto do **Instituto Federal do Rio Grande do Sul (IFRS)**.

*Pequenas escolhas geram grandes mudanças.* 🌿