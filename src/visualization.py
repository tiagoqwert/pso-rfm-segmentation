"""
visualization.py

MÓDULO DE VISUALIZAÇÃO DOS CLUSTERS (PSO)

Este arquivo concentra:
(1) Projeção PCA em 2D para visualização dos clusters
(2) Geração do gráfico com pontos e centróides

Observação:
- A projeção PCA é usada apenas para visualização (redução para 2D).
- A execução do PSO e geração dos dados são responsabilidade do main.py.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# VISUALIZAÇÃO COM PCA (2D)
# ============================================================

def plot_pca_clusters(X_scaled, labels, centroids):
    """
    Projeta os dados e centróides em 2D via PCA e gera um scatter plot.

    Parâmetros
    ----------
    X_scaled : np.ndarray
        Dados padronizados (n, d).
    labels : np.ndarray
        Rótulos de cluster para cada ponto.
    centroids : np.ndarray
        Centrôides no espaço padronizado (antes da projeção PCA).
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    cent_2d = pca.transform(centroids)

    plt.figure(figsize=(8, 6))

    # Pontos projetados em 2D, coloridos pelo cluster
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )

    # Centrôides projetados em 2D
    plt.scatter(
        cent_2d[:, 0],
        cent_2d[:, 1],
        c="black",
        marker="X",
        s=200,
        label="Centróides"
    )

    plt.title("Clusters obtidos pelo PSO (PCA 2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/clusters_pso_pca.png", dpi=150)
    print("\nImagem salva: clusters_pso_pca.png")