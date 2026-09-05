# Assistente de Resíduos Sólidos e Descarte Inteligente

Esqueleto inicial: RAG (texto) + reconhecimento de imagem (VLM) para
classificar resíduos e orientar o descarte correto.

## Estrutura

```
residuo-llm/
├── app/
│   ├── config.py     # configurações e categorias de resíduos
│   ├── vision.py      # classificação de imagem/frames de vídeo
│   ├── rag.py         # busca de contexto + geração de resposta
│   ├── ingest.py       # indexação dos documentos da base de conhecimento
│   └── main.py          # API FastAPI (endpoints /chat, /classify-image)
├── data/
│   ├── docs/           # coloque aqui seus .txt/.md (legislação, manuais)
│   └── chroma/         # banco vetorial persistente (gerado automaticamente)
├── requirements.txt
└── .env.example
```

## Como rodar

1. Crie o ambiente e instale dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure as variáveis de ambiente:
   ```bash
   cp .env.example .env
   # edite .env e coloque sua OPENAI_API_KEY
   ```

3. Adicione documentos reais em `data/docs/` (legislação, guias de coleta
   seletiva do seu município, manuais de reciclagem etc. em `.txt` ou `.md`)
   e indexe:
   ```bash
   python -m app.ingest
   ```

4. Suba a API:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Teste:
   ```bash
   # Pergunta em texto
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"pergunta": "como descarto pilhas?"}'

   # Classificação por imagem
   curl -X POST http://localhost:8000/classify-image \
     -F "imagem=@caminho/para/foto.jpg" \
     -F "pergunta=Como descarto isso?"
   ```

## Próximos passos sugeridos

- **Base de conhecimento**: trocar o `exemplo-pilhas-baterias.md` por
  documentos reais (legislação municipal/estadual, guias de coleta seletiva,
  lista de ecopontos com geolocalização).
- **Precisão visual**: se a classificação direta pelo VLM não for precisa o
  suficiente, treinar/fine-tunar um detector dedicado (YOLOv8 com datasets
  como TACO ou TrashNet) e usar sua saída como contexto extra no prompt de
  `vision.py`.
- **Vídeo**: implementar extração de frames (ffmpeg/opencv) antes de chamar
  `classify_video_frames()`.
- **Geolocalização**: integrar uma base de pontos de coleta/ecopontos para
  a resposta indicar o local mais próximo.
- **Frontend**: app mobile/web com câmera, chat e histórico.
- **Avaliação**: montar um conjunto de perguntas/imagens de teste para medir
  acurácia da classificação e qualidade das respostas do RAG.
