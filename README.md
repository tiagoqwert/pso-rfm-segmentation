# PSO para Segmentação de Clientes com Modelo RFM

Este trabalho implementa **Particle Swarm Optimization (PSO)** aplicado ao problema de **clustering de clientes** usando o modelo **RFM (Recency, Frequency, Monetary)**.

O objetivo é segmentar clientes em grupos com comportamentos semelhantes e analisar os perfis obtidos.

O projeto inclui:

- geração de dados RFM simulados
- padronização dos dados
- clustering baseado em PSO
- seleção do número de clusters (K)
- visualização dos clusters via PCA
- resumo estatístico e interpretação dos segmentos

---

## Modelo RFM

Cada cliente é representado por três atributos:

- **Recency**: dias desde a última compra (menor → mais recente)
- **Frequency**: número de compras (maior → mais frequente)
- **Monetary**: valor gasto (maior → mais valioso)

Essas variáveis são padronizadas antes do clustering.

---

## PSO para Clustering

Neste trabalho:

- cada partícula representa um conjunto de centróides
- o fitness é a **SSE (Sum of Squared Errors)**
- o objetivo é minimizar a dispersão intra-cluster
- clusters vazios recebem penalização

Para reduzir o efeito da aleatoriedade, a execução final repete o PSO com múltiplas seeds e seleciona a melhor solução.

---

## Estrutura do Projeto
 - data_and_fitness.py → geração de dados RFM e funções auxiliares
 - pso.py → implementação do PSO para clustering
 - k_selection.py → escolha do melhor número de clusters
 - visualization.py → visualização dos clusters via PCA
 - cluster_report.py → resumo e interpretação dos clusters
 - main.py → execução principal do experimento

 
---

## Como Executar

### 1) Selecionar o número de clusters (opcional)
 - python k_selection.py


### 2) Executar o experimento principal
 - python main.py


Isso irá:

- rodar PSO com múltiplas seeds
- salvar o gráfico PCA dos clusters
- imprimir o resumo RFM dos grupos
- exibir perfis interpretáveis dos clusters

---

## Saídas Geradas

- `clusters_pso_pca.png` → visualização dos clusters em PCA 2D
- tabela resumo dos clusters (tamanho e médias RFM)
- perfis interpretáveis dos segmentos

---

## Dependências

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
