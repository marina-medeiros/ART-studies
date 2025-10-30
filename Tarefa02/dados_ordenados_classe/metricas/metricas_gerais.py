def average_accuracy(num_tasks, acc_matrix):
  denominator = num_tasks*(num_tasks+1)/2
  acc_sum = 0

  for i in range(num_tasks):
    for j in range(i):
      acc_sum += acc_matrix[i][j]

  return (acc_sum / denominator)

def backward_transfer(num_tasks, acc_matrix):
  denominator = num_tasks*(num_tasks-1)/2
  acc_sum = 0

  for i in range(2, num_tasks):
    for j in range(i-1):
      acc_sum += (acc_matrix[i][j] - acc_matrix[j][j])

  return (acc_sum / denominator)

def forward_transfer(num_tasks, acc_matrix):
  denominator = num_tasks*(num_tasks-1)/2
  acc_sum = 0
  j = 0

  for i in range(j-1):
    for j in range(2, num_tasks):
      acc_sum = acc_sum + acc_matrix[i][j]

  return (acc_sum / denominator)