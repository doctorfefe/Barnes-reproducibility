import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances


class QuadTree:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.children = []
        self.points = []
        self.mass_center = np.zeros(2)
        self.mass = 0.0

    def insert(self, point, index):
        if not self._in_bounds(point):
            return False

        if len(self.points) == 0 and not self.children:
            self.points.append((point, index))
            self.mass = 1.0
            self.mass_center = point.copy()
            return True

        if not self.children:
            self._subdivide()
            for p, i in self.points:
                self._insert_into_children(p, i)
            self.points = []

        if self._insert_into_children(point, index):
            self.mass_center = (self.mass_center * self.mass + point) / (self.mass + 1)
            self.mass += 1
            return True
        return False

    def _insert_into_children(self, point, index):
        return any(child.insert(point, index) for child in self.children)

    def _subdivide(self):
        half = self.size / 2
        quarter = self.size / 4
        cx, cy = self.center
        offsets = [(-quarter, -quarter), (quarter, -quarter), 
                   (-quarter, quarter), (quarter, quarter)]
        self.children = [
            QuadTree(np.array([cx + dx, cy + dy]), half) for dx, dy in offsets
        ]

    def _in_bounds(self, point):
        return np.all(np.abs(point - self.center) <= self.size / 2)

    def compute_force(self, point, theta=0.5, eps=1e-8):
        force = np.zeros(2)
        if self.mass == 0 or (len(self.points) == 1 and np.allclose(self.points[0][0], point)):
            return force

        dist2 = np.sum((point - self.mass_center) ** 2) + eps
        width = self.size

        if width / np.sqrt(dist2) < theta or not self.children:
            force = self.mass * (point - self.mass_center) / ((1 + dist2) ** 2)
        else:
            for child in self.children:
                force += child.compute_force(point, theta, eps)
        return force


def compute_pairwise_affinities(X, perplexity=30.0, tol=1e-5):
    D = pairwise_distances(X, squared=True)
    n = X.shape[0]
    P = np.zeros((n, n))

    for i in range(n):
        idx = np.arange(n) != i  
        Di = D[i][idx]

        betamin, betamax = -np.inf, np.inf
        beta = 1.0
        H, thisP = _Hbeta(Di, beta)
        Hdiff = H - np.log(perplexity)
        tries = 0

        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
            H, thisP = _Hbeta(Di, beta)
            Hdiff = H - np.log(perplexity)
            tries += 1

        P[i, idx] = thisP

    P = (P + P.T) / (2 * n)
    return P

def _Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    sumP = max(sumP, 1e-8)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, P / sumP

def tsne_barnes_hut(X, perplexity=30.0, learning_rate=100.0, n_iter=500):
    n = X.shape[0]
    Y = np.random.randn(n, 2) * 1e-4
    P = compute_pairwise_affinities(X, perplexity=perplexity)
    P *= 4  

    for iter in range(n_iter):
        grads = np.zeros_like(Y)

        tree = QuadTree(center=np.mean(Y, axis=0), size=np.max(np.ptp(Y, axis=0)))
        for i in range(n):
            tree.insert(Y[i], i)

        for i in range(n):
            attractive = np.sum((Y[i] - Y) * P[i, :, None], axis=0)
            repulsive = tree.compute_force(Y[i])
            grads[i] = attractive - repulsive

        Y -= learning_rate * grads

        if iter == 100:
            P /= 4  

        if iter % 100 == 0 or iter == n_iter - 1:
            print(f"Iter {iter:03d}: ||grad|| = {np.linalg.norm(grads):.4f}")

    return Y

