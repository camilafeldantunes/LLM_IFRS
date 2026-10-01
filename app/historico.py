import sqlite3
from datetime import datetime
from pathlib import Path

# Salva em <raiz do projeto>/data/historico.db, independente de onde o app é executado
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "historico.db"


def _conectar():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            criado_em TEXT NOT NULL,
            origem TEXT,
            sessao_id TEXT,
            pergunta TEXT,
            tem_imagem INTEGER,
            objeto TEXT,
            categoria_visao TEXT,
            confianca TEXT,
            resposta TEXT,
            fontes TEXT
        )
        """
    )
    return conn


def registrar(pergunta, resposta, fontes=None, classificacao=None,
              tem_imagem=False, origem="streamlit", sessao_id=None):
    """Grava uma interação. Nunca levanta erro: falha no log não pode quebrar o app."""
    try:
        c = classificacao or {}
        with _conectar() as conn:
            conn.execute(
                """
                INSERT INTO interacoes
                (criado_em, origem, sessao_id, pergunta, tem_imagem, objeto,
                 categoria_visao, confianca, resposta, fontes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    origem,
                    sessao_id,
                    pergunta,
                    int(bool(tem_imagem)),
                    c.get("objeto"),
                    c.get("categoria"),
                    c.get("confianca"),
                    resposta,
                    ", ".join(fontes or []),
                ),
            )
    except Exception as e:
        print(f"[historico] não foi possível salvar: {e}")