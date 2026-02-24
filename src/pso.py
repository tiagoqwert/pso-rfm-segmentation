"""
pso.py

MÓDULO DE PSO PARA CLUSTERING (K FIXO)

Este arquivo implementa um PSO "básico" aplicado ao problema de clustering:

Representação
-------------
- Cada partícula representa um conjunto de centróides:
  partícula = matriz (k, d), onde:
    k = número de clusters
    d = número de dimensões (features)

Função objetivo (fitness)
-------------------------
- SSE (Sum of Squared Errors):
  soma das distâncias quadráticas de cada ponto ao centróide mais próximo.
- Quanto menor o SSE, melhor (clusters mais compactos).

Saídas
------
- Melhores centróides globais (gbest)
- Melhor fitness global (SSE)
- Labels finais (cluster de cada amostra)
"""

import numpy as np

from src.data_and_fitness import (
    inicializar_centroides_aleatorios,
    sse_fitness,
    atribuir_clusters
)


# ============================================================
# 1) PSO PARA OTIMIZAÇÃO DOS CENTRÓIDES
# ============================================================

def pso_clustering(
    X_scaled,
    k: int,
    n_particles: int = 30,
    n_iters: int = 80,
    w: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    seed: int = 42,
):
    """
    Executa PSO para otimizar centróides de clustering (K fixo).

    Hiperparâmetros do PSO
    ----------------------
    w :
        Peso de inércia (equilíbrio entre exploração e explotação).
    c1 :
        Componente cognitiva (atração para o melhor da própria partícula).
    c2 :
        Componente social (atração para o melhor global do enxame).

    Parâmetros
    ----------
    X_scaled : np.ndarray
        Dados padronizados (n, d).
    k : int
        Número de clusters (fixo).
    n_particles : int
        Número de partículas (tamanho do enxame).
    n_iters : int
        Número de iterações do PSO.
    w, c1, c2 : float
        Parâmetros do PSO.
    seed : int
        Semente de aleatoriedade.

    Retorno
    -------
    gbest_centroids : np.ndarray (k, d)
        Melhor conjunto de centróides encontrado.
    gbest_fitness : float
        Melhor SSE (a ser minimizado).
    labels : np.ndarray (n,)
        Rótulo de cluster para cada amostra.
    """
    rng = np.random.default_rng(seed)
    n_samples, d = X_scaled.shape

    # Limites por dimensão no espaço padronizado (para clipping)
    X_min = X_scaled.min(axis=0)
    X_max = X_scaled.max(axis=0)

    # ============================
    # Inicialização do enxame
    # ============================
    positions = np.zeros((n_particles, k, d))
    velocities = np.zeros((n_particles, k, d))

    for i in range(n_particles):
        # Inicialização dos centróides usando pontos do próprio dataset
        positions[i] = inicializar_centroides_aleatorios(
            X_scaled,
            k=k,
            seed=int(rng.integers(0, 1_000_000))
        )

        # Velocidades iniciais pequenas (evita saltos muito grandes no começo)
        velocities[i] = rng.normal(0, 0.1, size=(k, d))

    pbest_pos = positions.copy()
    pbest_fit = np.array([sse_fitness(X_scaled, positions[i]) for i in range(n_particles)])

    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    for _ in range(n_iters):
        r1 = rng.random(size=(n_particles, k, d))
        r2 = rng.random(size=(n_particles, k, d))

        # Atualiza velocidades (inércia + termos cognitivo e social)
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos[None, :, :] - positions)
        )

        positions = positions + velocities

        positions = np.clip(positions, X_min[None, None, :], X_max[None, None, :])

        # Avaliação e atualização de pbest / gbest
        for i in range(n_particles):
            fit = sse_fitness(X_scaled, positions[i])

            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i].copy()

                if fit < gbest_fit:
                    gbest_fit = float(fit)
                    gbest_pos = positions[i].copy()

    # Labels finais usando o melhor conjunto de centróides (gbest)
    labels = atribuir_clusters(X_scaled, gbest_pos)

    return gbest_pos, gbest_fit, labels


# ============================================================
# 2) TESTE RÁPIDO DO MÓDULO
# ============================================================

if __name__ == "__main__":
    from data_and_fitness import simular_dataset_rfm, preparar_dados

    df = simular_dataset_rfm(n=300, seed=42)
    X_scaled, _ = preparar_dados(df)

    k = 4
    centroids, best_fit, labels = pso_clustering(X_scaled, k=k)

    print("PSO clustering (teste)")
    print("K =", k)
    print("Best fitness (SSE):", round(best_fit, 4))
    print("Distribuição de clusters:", np.bincount(labels))
