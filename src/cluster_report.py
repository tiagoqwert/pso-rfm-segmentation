"""
cluster_report.py

FUNÇÕES DE RESUMO E INTERPRETAÇÃO DOS CLUSTERS

Este módulo concentra:
(1) Tabela resumo por cluster (N, percentual, médias RFM)
(2) Heurística simples de perfil por cluster (comparação com médias globais)
"""

import pandas as pd


def resumir_clusters(df, labels):
    """
    Monta um resumo por cluster contendo:
    - N (tamanho do grupo)
    - Percentual no dataset
    - Médias de Recency, Frequency e Monetary
    """
    df2 = df.copy()
    df2["cluster"] = labels

    tamanhos = df2["cluster"].value_counts().sort_index()
    medias = df2.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

    resumo = medias.copy()
    resumo["N"] = tamanhos
    resumo["Percentual"] = (tamanhos / len(df2) * 100).round(2)

    resumo = resumo[["N", "Percentual", "Recency", "Frequency", "Monetary"]]
    return resumo


def interpretar(resumo):
    """
    Gera rótulos textuais por cluster comparando (R, F, M) com as médias globais.
    """
    mean_r = resumo["Recency"].mean()
    mean_f = resumo["Frequency"].mean()
    mean_m = resumo["Monetary"].mean()

    perfis = {}
    for c, row in resumo.iterrows():
        tags = []
        tags.append("recente" if row["Recency"] < mean_r else "pouco recente")
        tags.append("alta frequência" if row["Frequency"] > mean_f else "baixa frequência")
        tags.append("alto gasto" if row["Monetary"] > mean_m else "baixo gasto")
        perfis[c] = ", ".join(tags)

    return perfis


def imprimir_resumo_e_perfis(df, labels, ndigits=3):
    """
    Imprime:
    - tabela resumo
    - perfis sugeridos
    """
    resumo = resumir_clusters(df, labels)

    print("\n===== RESUMO DOS CLUSTERS (RFM) =====")
    print(resumo.round(ndigits))

    perfis = interpretar(resumo)

    print("\n===== PERFIS SUGERIDOS =====")
    for c in sorted(perfis.keys()):
        print(f"Cluster {c}: {perfis[c]}")

    return resumo, perfis