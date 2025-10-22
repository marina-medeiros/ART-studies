from artlib import FuzzyART, FuzzyARTMAP, FusionART
from sklearn.metrics import classification_report, adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
  
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

  return (acc_sum / denominator)
