"""
data_and_fitness.py

MÓDULO BASE DO PROJETO (RFM + funções auxiliares)

Este arquivo concentra:
(1) Geração do dataset RFM (simulado) para testes/reprodutibilidade
(2) Pré-processamento (padronização com StandardScaler)
(3) Funções básicas usadas pelo PSO para clustering:
    - inicialização de centróides
    - atribuição de clusters
    - função fitness (SSE/inércia)

Observação:
- Os "perfis" A/B/C/D aqui são apenas para SIMULAR dados mais realistas.
- O algoritmo (PSO) NÃO recebe esses rótulos: ele só vê os pontos (R,F,M).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1) DATASET (RFM SIMULADO)
# ============================================================

def simular_dataset_rfm(n=300, seed=42):
    """
    Gera um dataset simulado de clientes com variáveis RFM:
    - Recency   : dias desde a última compra (quanto menor, mais recente/ativo)
    - Frequency : número de compras no período (quanto maior, mais frequente)
    - Monetary  : valor monetário associado ao cliente (quanto maior, mais valioso)

    Parâmetros
    ----------
    n : int
        Número de clientes simulados.
    seed : int
        Semente para reprodutibilidade.

    Retorno
    -------
    df : pandas.DataFrame
        DataFrame com colunas ["Recency", "Frequency", "Monetary"].
    """
    rng = np.random.default_rng(seed)

    # Dividimos os clientes em 4 perfis para criar dados "com estrutura"
    # (isso facilita obter clusters interpretáveis em experimentos)
    n1 = n // 4
    n2 = n // 4
    n3 = n // 4
    n4 = n - (n1 + n2 + n3)

    # Perfil A: frequente e alto gasto (clientes premium frequentes)
    rec_a = rng.normal(10, 5, n1)
    freq_a = rng.normal(10, 3, n1)
    mon_a = rng.normal(800, 150, n1)

    # Perfil B: frequente e baixo gasto (muito ativo, mas ticket baixo)
    rec_b = rng.normal(12, 6, n2)
    freq_b = rng.normal(12, 4, n2)
    mon_b = rng.normal(120, 40, n2)

    # Perfil C: pouco frequente e alto gasto (premium ocasional)
    rec_c = rng.normal(35, 10, n3)
    freq_c = rng.normal(2, 1, n3)
    mon_c = rng.normal(900, 200, n3)

    # Perfil D: pouco frequente e gasto médio (baixo engajamento / risco de churn)
    rec_d = rng.normal(40, 12, n4)
    freq_d = rng.normal(3, 1.5, n4)
    mon_d = rng.normal(300, 80, n4)

    # Concatena todos os perfis em um único conjunto
    rec = np.concatenate([rec_a, rec_b, rec_c, rec_d])
    freq = np.concatenate([freq_a, freq_b, freq_c, freq_d])
    mon = np.concatenate([mon_a, mon_b, mon_c, mon_d])

    # Garante valores mínimos plausíveis (evita recency/frequency negativos)
    df = pd.DataFrame({
        "Recency": np.clip(rec, 1, None),
        "Frequency": np.clip(freq, 1, None),
        "Monetary": np.clip(mon, 10, None),
    })

    return df


# ============================================================
# 2) PRÉ-PROCESSAMENTO
# ============================================================

def preparar_dados(df, colunas=None):
    """
    Seleciona colunas do DataFrame e aplica padronização (StandardScaler).

    Por que padronizar?
    - Como R, F e M estão em escalas diferentes, a distância Euclidiana
      seria dominada pela variável com maior magnitude (geralmente Monetary).
    - O scaler z-score (média 0, desvio 1) torna as variáveis comparáveis.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Dados com colunas RFM.
    colunas : list[str] ou None
        Lista das colunas usadas como features. Se None, usa RFM padrão.

    Retorno
    -------
    X_scaled : np.ndarray (n, d)
        Matriz padronizada.
    scaler : StandardScaler
        Objeto do scaler (útil caso você queira desfazer a escala depois).
    """
    if colunas is None:
        colunas = ["Recency", "Frequency", "Monetary"]

    # Matriz de features (n amostras x d atributos)
    X = df[colunas].values

    # Padronização z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ============================================================
# 3) FUNÇÕES AUXILIARES PARA CLUSTERING
# ============================================================

def inicializar_centroides_aleatorios(X_scaled, k, seed=0):
    """
    Inicializa centróides escolhendo K pontos aleatórios do próprio dataset.

    Isso é uma inicialização simples e costuma evitar centróides fora da nuvem
    (diferente de chutar valores arbitrários).

    Parâmetros
    ----------
    X_scaled : np.ndarray
        Dados padronizados (n, d).
    k : int
        Número de clusters.
    seed : int
        Semente de aleatoriedade.

    Retorno
    -------
    centroides : np.ndarray (k, d)
        Centrôides iniciais.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_scaled), size=k, replace=False)
    return X_scaled[idx].copy()


def atribuir_clusters(X_scaled, centroides):
    """
    Atribui cada ponto ao centróide mais próximo (distância Euclidiana).

    Implementação vetorizada:
    - cria matriz (n, k, d) de diferenças
    - calcula norma (n, k) das distâncias
    - pega argmin para obter o rótulo de cada ponto

    Retorno
    -------
    labels : np.ndarray (n,)
        Rótulo do cluster para cada ponto.
    """
    dists = np.linalg.norm(
        X_scaled[:, None, :] - centroides[None, :, :],
        axis=2
    )
    labels = np.argmin(dists, axis=1)
    return labels


def sse_fitness(X_scaled, centroides):
    """
    Fitness do clustering (função objetivo do PSO):
    SSE = soma dos erros quadráticos intra-cluster (inércia).

    - Para cada cluster j, soma ||x - c_j||^2 dos pontos atribuídos.
    - Quanto menor o SSE, melhor (clusters mais compactos).

    Observação importante:
    - Se um cluster ficar vazio, aplicamos penalização alta,
      para evitar soluções degeneradas durante a otimização.

    Retorno
    -------
    sse : float
        Valor de SSE (a ser minimizado).
    """
    labels = atribuir_clusters(X_scaled, centroides)

    sse = 0.0
    for j in range(len(centroides)):
        pts = X_scaled[labels == j]

        if len(pts) == 0:
            # Penalização para cluster vazio (evita "soluções ruins" no PSO)
            sse += 1e6
        else:
            dif = pts - centroides[j]
            sse += np.sum(dif ** 2)

    return sse


# ============================================================
# 4) TESTE RÁPIDO DO MÓDULO
# ============================================================

if __name__ == "__main__":
    # Teste rápido para verificar:
    # - geração do dataset
    # - padronização
    # - SSE para uma inicialização aleatória
    print("Teste do módulo data_and_fitness.py\n")

    df = simular_dataset_rfm(n=200, seed=42)
    X_scaled, scaler = preparar_dados(df)

    k = 4
    centroides = inicializar_centroides_aleatorios(X_scaled, k=k, seed=0)

    fitness = sse_fitness(X_scaled, centroides)

    print("Shape dos dados:", X_scaled.shape)
    print("Fitness inicial (SSE):", round(fitness, 4))