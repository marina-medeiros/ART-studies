from artlib import FuzzyARTMAP
from sklearn.metrics import accuracy_score
import numpy as np
import torch

# # Os limites são calculados de forma diferente quando se trata de imagens
def train_fuzzyARTMAP_images(X_train_subset, y_train_subset, X_test_subset, y_test_subset, n_dim):
    fuzzy_artmap_model = FuzzyARTMAP(rho=0.3, alpha=0.0, beta=1.0)

    # 🔧 Conversão para numpy
    X_train_subset = X_train_subset.numpy() if isinstance(X_train_subset, torch.Tensor) else X_train_subset
    y_train_subset = y_train_subset.numpy() if isinstance(y_train_subset, torch.Tensor) else y_train_subset
    X_test_subset = X_test_subset.numpy() if isinstance(X_test_subset, torch.Tensor) else X_test_subset
    y_test_subset = y_test_subset.numpy() if isinstance(y_test_subset, torch.Tensor) else y_test_subset


    lower_bounds = np.zeros(n_dim)
    upper_bounds = np.full(n_dim, 255.0)
    fuzzy_artmap_model.module_a.set_data_bounds(lower_bounds, upper_bounds)

    train_X_fuzzy_artmap = fuzzy_artmap_model.prepare_data(X_train_subset)
    test_X_fuzzy_artmap  = fuzzy_artmap_model.prepare_data(X_test_subset)

    fuzzy_artmap_model.partial_fit(train_X_fuzzy_artmap, y_train_subset)
    fuzzy_artmap_predictions = fuzzy_artmap_model.predict(test_X_fuzzy_artmap)

    return accuracy_score(y_test_subset,fuzzy_artmap_predictions)

def train_fuzzyARTMAP(X_train_subset, y_train_subset, X_test_subset, y_test_subset):
    fuzzy_artmap_model = FuzzyARTMAP(rho=0.3, alpha=0.0, beta=1.0)

    X_combined = np.concatenate([X_train_subset, X_test_subset], axis=0)
    lower_bound, upper_bound = fuzzy_artmap_model.find_data_bounds(X_combined)
    fuzzy_artmap_model.module_a.set_data_bounds(lower_bound, upper_bound)

    train_X_fuzzy_artmap = fuzzy_artmap_model.prepare_data(X_train_subset)
    test_X_fuzzy_artmap  = fuzzy_artmap_model.prepare_data(X_test_subset)

    fuzzy_artmap_model.partial_fit(train_X_fuzzy_artmap, y_train_subset)
    fuzzy_artmap_predictions = fuzzy_artmap_model.predict(test_X_fuzzy_artmap)

    return accuracy_score(y_test_subset,fuzzy_artmap_predictions)


def generate_acc_matrix_fuzzyARTMAP(num_tasks, X_train_sorted, y_train_sorted, X_test_sorted, y_test_sorted, images):
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

            if X_train_subset.shape[0] == 0 or X_test_subset.shape[0] == 0:
                acc_matrix[i][j] = 0.0 # ou 0.0, dependendo do que quiser preencher
                continue

            if(images):
                acc_matrix[i][j] = train_fuzzyARTMAP_images(
                                    X_train_subset,
                                    y_train_subset,  # não é usado na função
                                    X_test_subset,
                                    y_test_subset, 
                                    16 * 16)
            else:
                acc_matrix[i][j] = train_fuzzyARTMAP(
                                    X_train_subset,
                                    y_train_subset,  # não é usado na função
                                    X_test_subset,
                                    y_test_subset)
    return acc_matrix