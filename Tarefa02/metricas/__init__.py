# my_package/__init__.py

# 1. Define package-level variables
__version__ = "1.0.0"
__author__ = "marina-medeiros"

# 2. Import specific functions/classes directly into the package namespace
# This allows users to import 'say_hello' directly from 'my_package'
# instead of 'my_package.greetings'
from .metricas_fuzzyART import train_fuzzyART, train_fuzzyART_images, generate_acc_matrix_fuzzyART

from .metricas_fuzzyARTMAP import train_fuzzyARTMAP, train_fuzzyARTMAP_images, generate_acc_matrix_fuzzyARTMAP

from .metricas_gerais import average_accuracy, forward_transfer, backward_transfer

# 3. Define __all__ to control 'from my_package import *' behavior
# In this case, only 'say_hello' and '__version__' would be imported
# if a user uses 'from my_package import *'
__all__ = ["average_accuracy", "forward_transfer", "backward_transfer", 
           "generate_acc_matrix_fuzzyARTMAP",
           "generate_acc_matrix_fuzzyART"]

# 4. Execute code upon package import (e.g., for setup or logging)
print(f"Initializing my_package version {__version__} by {__author__}")