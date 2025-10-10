# Métricas para a avaliação do aprendizado contínuo
Não basta observar a acurácia fianl de um algoritmo. É necessário avaliar o quão rápido ele aprende e esquece, se ele consgue tranferir conhecimento de uma tarefa para outra e se o algoritmo é estável e eficiente enquanto aprende.

## Performance geral:
 $\Omega = \frac{1}{N} \sum_{i = 1}^{N} \frac{R_{i,j}}{R_{i,j}^C}$

Aqui, $R_{i,j}^C$ representa a melhro acurácia possível que poderíamos obter com um conjunto de testes caso o modelo fosse treinado como todos os dados de uma vez.

> Observação: $Tr_i$ é a coleção dos dados de treino em um momento i.

Quando $\Omega = 1$, há a indição de que a performance teve desempenho idêntico a uma configuração cumulativa off-line.

Se $\Omega \gt 1$, é possível que o modelo offline seja pior do que o modelo treinado no paradigma do Aprendizado Contínuo.


## Forgetting Ratio Metric

## Average Accuracy (ACC)

## Backward Transfer (BWT)

## Forward Transfer (FWT)