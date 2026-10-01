"""Exporta o histórico para CSV (abre direto no Excel/Google Planilhas).

Uso (a partir da raiz do projeto):  python app/exportar_historico.py
"""
import csv
import sqlite3

from historico import DB_PATH

SAIDA = DB_PATH.with_name("historico.csv")

if not DB_PATH.exists():
    raise SystemExit("Ainda não há histórico salvo.")

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.execute("SELECT * FROM interacoes ORDER BY id")
    colunas = [d[0] for d in cur.description]
    linhas = cur.fetchall()

with open(SAIDA, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(colunas)
    w.writerows(linhas)

print(f"{len(linhas)} interações exportadas para {SAIDA}")