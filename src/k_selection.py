"""
k_selection.py

MÓDULO DE SELEÇÃO DO NÚMERO DE CLUSTERS (K) USANDO PSO

Este arquivo concentra:
(1) Execução do PSO para múltiplos valores de K
(2) Avaliação com métricas internas de clustering:
    - Silhouette (maior é melhor)
    - Davies-Bouldin Index (menor é melhor)
    - SSE (inércia intra-cluster)
    - Tempo de execução
(3) Escolha automática do melhor K
(4) Geração de gráficos e impressão tabular dos resultados

Observação:
- O PSO é executado separadamente para cada valor de K.
- A escolha final prioriza Silhouette e usa DBI como critério auxiliar.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.data_and_fitness import simular_dataset_rfm, preparar_dados, sse_fitness
from src.pso import pso_clustering


# ============================================================
# 1) BUSCA DO MELHOR K COM PSO
# ============================================================

def escolher_k_com_pso(
    X_scaled,
    k_min=2,
    k_max=6,
    seed=42,
    pso_particles=30,
    pso_iters=80,
):
    """
    Executa o PSO para K variando de k_min até k_max e coleta métricas
    de qualidade para cada configuração.

    Métricas avaliadas
    ------------------
    Silhouette :
        Mede separação e coesão dos clusters.
    Davies-Bouldin (DBI) :
        Mede similaridade entre clusters.
    SSE :
        Soma dos erros quadráticos intra-cluster.
    Tempo :
        Tempo de execução do PSO.

    Estratégia de escolha
    ---------------------
    1) Maior Silhouette
    2) Em caso de empate (ou valores próximos), menor DBI

    Retorno
    -------
    melhor_k : int
        Valor de K escolhido automaticamente.
    resultados : list[dict]
        Lista contendo métricas coletadas para cada K.
    """
    resultados = []

    # Executa o PSO para cada valor de K
    for k in range(k_min, k_max + 1):
        start = time.time()

        centroids, best_fit, labels = pso_clustering(
            X_scaled,
            k=k,
            n_particles=pso_particles,
            n_iters=pso_iters,
            seed=seed,
        )

        elapsed = time.time() - start

        # Métricas internas de validação
        sil = silhouette_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)
        sse = sse_fitness(X_scaled, centroids)

        resultados.append({
            "k": k,
            "silhouette": float(sil),
            "dbi": float(dbi),
            "sse": float(sse),
            "time": float(elapsed),
        })

        print(f"K={k} | Sil={sil:.4f} | DBI={dbi:.4f} | SSE={sse:.4f} | t={elapsed:.3f}s")

    # Ordenação conforme critério definido
    resultados_sorted = sorted(resultados, key=lambda r: (-r["silhouette"], r["dbi"]))
    melhor_k = resultados_sorted[0]["k"]

    return melhor_k, resultados


# ============================================================
# 2) GERAÇÃO DE GRÁFICOS
# ============================================================

def salvar_graficos(resultados):
    """
    Gera gráficos das métricas em função de K.
    """
    ks = [r["k"] for r in resultados]
    sils = [r["silhouette"] for r in resultados]
    dbis = [r["dbi"] for r in resultados]

    # Silhouette por K
    plt.figure(figsize=(7, 5))
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette por K (PSO) - maior é melhor")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.grid(True)
    plt.savefig("outputs/silhouette_por_k.png", dpi=150)
    print("Imagem salva: silhouette_por_k.png")

    # DBI por K
    plt.figure(figsize=(7, 5))
    plt.plot(ks, dbis, marker="o")
    plt.title("Davies-Bouldin por K (PSO) - menor é melhor")
    plt.xlabel("K")
    plt.ylabel("DBI")
    plt.grid(True)
    plt.savefig("outputs/dbi_por_k.png", dpi=150)
    print("Imagem salva: dbi_por_k.png")


# ============================================================
# 3) IMPRESSÃO TABULAR DOS RESULTADOS
# ============================================================

def imprimir_tabela(resultados):
    """
    Exibe tabela formatada com as métricas obtidas para cada K.
    """
    print("\n================ RESULTADOS (PSO) ================")
    print(f"{'K':>2} | {'Silhouette':>10} | {'DBI':>8} | {'SSE':>10} | {'Tempo(s)':>8}")
    print("-" * 56)
    for r in resultados:
        print(f"{r['k']:>2} | {r['silhouette']:>10.4f} | {r['dbi']:>8.4f} | {r['sse']:>10.4f} | {r['time']:>8.3f}")
    print("==================================================\n")


# ============================================================
# 4) TESTE DO MÓDULO
# ============================================================

if __name__ == "__main__":
    # Teste completo do processo de seleção de K

    # 1) Dados (RFM)
    df = simular_dataset_rfm(n=300, seed=42)
    X_scaled, _ = preparar_dados(df)

    # 2) Testar Ks
    melhor_k, resultados = escolher_k_com_pso(
        X_scaled,
        k_min=2,
        k_max=6,
        seed=42,
        pso_particles=30,
        pso_iters=80
    )

    # 3) Mostrar e salvar
    imprimir_tabela(resultados)
    salvar_graficos(resultados)

    print(f" Melhor K sugerido (critério: maior Silhouette, menor DBI): {melhor_k}")