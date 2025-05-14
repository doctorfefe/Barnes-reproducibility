import numpy as np
from barneshut import tsne_barnes_hut  
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
X, y = load_digits(return_X_y=True)
X, y = X[:1000], y[:1000]

Y = tsne_barnes_hut(X, perplexity=30.0, learning_rate=100.0, n_iter=500)

plt.figure(figsize=(10, 8))
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap='tab10', s=20)
plt.title("t-SNE con Barnes-Hut (Quadtree desde cero)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.show()
