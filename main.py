import numpy as np
import matplotlib.pyplot as plt
from tsne import compute_pairwise_affinities, initialize_embedding, gradient_descent_bh

# Cargar datos de ejemplo (sintéticos)
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X = X[:1000]  # Subconjunto para velocidad

# Paso 1: calcular probabilidades en alta dimensión
P = compute_pairwise_affinities(X, perplexity=30)

# Paso 2: inicializar en 2D
Y = initialize_embedding(X.shape[0])

# Paso 3: optimizar con Barnes-Hut
Y = gradient_descent_bh(Y, P, n_iter=500)

# Visualizar
plt.scatter(Y[:, 0], Y[:, 1], c=y[:1000], cmap='tab10')
plt.title("Barnes-Hut t-SNE desde cero")
plt.show()