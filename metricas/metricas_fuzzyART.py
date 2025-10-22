from artlib import FuzzyART, FuzzyARTMAP, FusionART
from sklearn.metrics import classification_report, adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch

# Os limites são calculados de forma diferente quando se trata de imagens
def train_fuzzyART_images(X_train_subset, y_train_subset, X_test_subset, y_test_subset, n_dim):
    fuzzy_art_model = FuzzyART(rho=0.3, alpha=0.0, beta=1.0)

    lower_bounds = np.zeros(n_dim)
    upper_bounds = np.full(n_dim, 255.0)
    fuzzy_art_model.set_data_bounds(lower_bounds, upper_bounds)

    train_X_fuzzy_art = fuzzy_art_model.prepare_data(X_train_subset)
    test_X_fuzzy_art  = fuzzy_art_model.prepare_data(X_test_subset)

    fuzzy_art_model.fit(train_X_fuzzy_art)
    fuzzy_art_predictions = fuzzy_art_model.predict(test_X_fuzzy_art)

    return adjusted_rand_score(y_test_subset,fuzzy_art_predictions)

def train_fuzzyART(X_train_subset, y_train_subset, X_test_subset, y_test_subset):
    fuzzy_art_model = FuzzyART(rho=0.3, alpha=0.0, beta=1.0)

    lower_bound, upper_bound = fuzzy_art_model.find_data_bounds(X)
    fuzzy_art_model.set_data_bounds(lower_bound, upper_bound)

    train_X_fuzzy_art = fuzzy_art_model.prepare_data(X_train_subset)
    test_X_fuzzy_art  = fuzzy_art_model.prepare_data(X_test_subset)

    fuzzy_art_model.fit(train_X_fuzzy_art)
    fuzzy_art_predictions = fuzzy_art_model.predict(test_X_fuzzy_art)

    return adjusted_rand_score(y_test_subset,fuzzy_art_predictions)

def generate_acc_matrix_fuzzyART(num_tasks, X_train_sorted, y_train_sorted, X_test_sorted, y_test_sorted, images):
  train_subsets = []
  test_subsets = []

  acc_matrix = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]

  for i in range(num_tasks):
    for j in range(num_tasks):
        # Classes até a i-ésima (inclusive)
        train_classes = torch.arange(0, i + 1)

        # Máscara de seleção para treino: todas as classes <= i
        mask_train = torch.isin(y_train_sorted, train_classes)
        X_train_subset = X_train_sorted[mask_train]
        y_train_subset = y_train_sorted[mask_train]

        # Máscara de seleção para teste: apenas a classe j
        mask_test = (y_test_sorted == j)
        X_test_subset = X_test_sorted[mask_test]
        y_test_subset = y_test_sorted[mask_test]

        # Armazena os subconjuntos (opcional)
        train_subsets.append((X_train_subset, y_train_subset))
        test_subsets.append((X_test_subset, y_test_subset))

        if(images):
            acc_matrix[i][j] = train_fuzzyART_images(
                                X_train_subset,
                                y_train_subset,  # não é usado na função
                                X_test_subset,
                                y_test_subset, 
                                16 * 16)
        else:
            acc_matrix[i][j] = train_fuzzyART(
                                X_train_subset,
                                y_train_subset,  # não é usado na função
                                X_test_subset,
                                y_test_subset)
    return acc_matrix
  
def average_accuracy(num_tasks, acc_matrix):
  denominator = num_tasks*(num_tasks+1)/2
  acc_sum = 0

  for i in range(num_tasks+1):
    for j in range(i+1):
      acc_sum = acc_sum + acc_matrix[i][j]

  return (acc_sum / denominator)

def backward_transfer(num_tasks, acc_matrix):
  denominator = num_tasks*(num_tasks-1)/2
  acc_sum = 0

  for i in range(2, num_tasks+1):
    for j in range(i):
      acc_sum = acc_sum + (acc_matrix[i][j] - acc_matrix[j][j])

  return (acc_sum / denominator)

def forward_transfer(num_tasks, acc_matrix):
  denominator = num_tasks*(num_tasks-1)/2
  acc_sum = 0
  j = 0

  for i in range(j-1):
    for j in range(num_tasks+1):
      acc_sum = acc_sum + acc_matrix[i][j]
