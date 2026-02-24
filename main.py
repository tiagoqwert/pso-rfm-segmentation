"""
main.py

MÓDULO PRINCIPAL (EXECUÇÃO COMPLETA DO PSO + RESUMO DOS CLUSTERS)

Este arquivo concentra:
(1) Geração do dataset RFM (simulado) e pré-processamento (padronização)
(2) Execução do PSO para clustering com K fixo
(3) Estratégia de robustez: repetir PSO com múltiplas seeds e escolher a melhor execução (menor SSE)
(4) Visualização dos clusters via PCA 2D
(5) Resumo estatístico (tamanho e médias RFM) e perfis interpretáveis por cluster

Observação:
- A interpretação é heurística: compara cada cluster com a média global (R, F, M).
- O objetivo aqui é apoiar análise e apresentação dos resultados.
"""

import numpy as np

from src.data_and_fitness import simular_dataset_rfm, preparar_dados
from src.pso import pso_clustering
from src.visualization import plot_pca_clusters
from src.cluster_report import imprimir_resumo_e_perfis



# ============================================================
# 1) PSO (MÚLTIPLAS SEEDS)
# ============================================================

def rodar_pso_varias_vezes(X_scaled, k, seeds, n_particles=30, n_iters=80):
    """
    Executa o PSO para várias seeds e seleciona a melhor execução (menor SSE).

    Motivação
    ---------
    - Como PSO envolve aleatoriedade (inicialização e termos r1/r2),
      diferentes seeds podem levar a soluções diferentes.
    - Repetir execuções reduz risco de escolher uma solução ruim por acaso.

    Parâmetros
    ----------
    X_scaled : np.ndarray
        Dados padronizados.
    k : int
        Número de clusters.
    seeds : list[int]
        Lista de seeds testadas.
    n_particles : int
        Número de partículas do PSO.
    n_iters : int
        Número de iterações do PSO.

    Retorno
    -------
    best : tuple
        (best_fit, seed, centroids, labels) da melhor execução.
    """
    best = None  # (best_fit, seed, centroids, labels)

    print(f"\nRodando PSO {len(seeds)} vezes para reduzir aleatoriedade...")

    for s in seeds:
        centroids, best_fit, labels = pso_clustering(
            X_scaled,
            k=k,
            n_particles=n_particles,
            n_iters=n_iters,
            seed=s
        )
        print(f"  seed={s} -> SSE={best_fit:.4f}")

        if best is None or best_fit < best[0]:
            best = (best_fit, s, centroids, labels)

    return best  # (best_fit, seed, centroids, labels)


# ============================================================
# 2) EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Pipeline principal:
    # (1) gerar dados RFM
    # (2) padronizar
    # (3) executar PSO com K fixo e múltiplas seeds
    # (4) gerar gráfico PCA
    # (5) imprimir resumo e perfis

    # 1) Dataset
    df = simular_dataset_rfm(n=300, seed=42)
    X_scaled, _ = preparar_dados(df)

    # 2) K escolhido via seleção (validado em experimentos)
    k = 4

    # 3) PSO robusto: múltiplas execuções com seeds diferentes
    seeds = [10, 20, 30, 40, 50]  # pode aumentar a lista para mais repetições
    best_fit, best_seed, centroids, labels = rodar_pso_varias_vezes(
        X_scaled, k=k, seeds=seeds, n_particles=30, n_iters=80
    )

    print(f"\n Melhor execução: seed={best_seed} com SSE={best_fit:.4f}")
    print("Distribuição:", np.bincount(labels))

    # 4) Gráfico PCA (salvo em arquivo)
    plot_pca_clusters(X_scaled, labels, centroids)

    # 5) Resumo e perfis
    imprimir_resumo_e_perfis(df, labels)