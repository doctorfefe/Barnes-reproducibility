import numpy as np
from scipy.spatial.distance import pdist, squareform

# Paso 1: Probabilidades en alta dimensión (P matrix)
def compute_pairwise_affinities(X, perplexity=30.0, tol=1e-5):
    n = X.shape[0]
    distances = squareform(pdist(X, 'euclidean')) ** 2
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_perp = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = _Hbeta(Di, beta[i])
        Hdiff = H - log_perp
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                beta[i] = (beta[i] * 2) if betamax == np.inf else (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                beta[i] = (beta[i] / 2) if betamin == -np.inf else (beta[i] + betamin) / 2
            (H, thisP) = _Hbeta(Di, beta[i])
            Hdiff = H - log_perp
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    P = (P + P.T) / (2 * n)
    return P

def _Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    if sumP < 1e-10:  # Evita división por cero
        P = np.ones_like(P) / len(P)
        H = np.log(len(P))
        return H, P
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


# Paso 2: Inicializar
def initialize_embedding(n, dim=2):
    return 1e-4 * np.random.randn(n, dim)

# Paso 3: QuadTree
class QuadTreeNode:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.mass = 0
        self.mass_center = np.zeros(2)
        self.point = None
        self.children = []

    def insert(self, point, index):
        if self.point is None and len(self.children) == 0:
            self.point = (point, index)
            self.mass = 1
            self.mass_center = point.copy()
        else:
            if len(self.children) == 0:
                self._subdivide()
                self._insert_into_children(self.point[0], self.point[1])
                self.point = None
            self._insert_into_children(point, index)
            self.mass += 1
            self.mass_center = (self.mass_center * (self.mass - 1) + point) / self.mass

    def _subdivide(self):
        x, y = self.center
        s = self.size / 2
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                child_center = np.array([x + dx * s / 2, y + dy * s / 2])
                self.children.append(QuadTreeNode(child_center, s))

    def _insert_into_children(self, point, index):
        for child in self.children:
            if child._contains(point):
                child.insert(point, index)
                return

    def _contains(self, point):
        return np.all(np.abs(point - self.center) <= self.size / 2)

def build_quadtree(Y):
    min_coord = np.min(Y, axis=0)
    max_coord = np.max(Y, axis=0)
    center = (min_coord + max_coord) / 2
    size = np.max(max_coord - min_coord) * 1.1
    root = QuadTreeNode(center, size)
    for i in range(Y.shape[0]):
        root.insert(Y[i], i)
    return root

def compute_repulsion_force(node, point, theta=0.5, eps=1e-5):
    def recurse(n):
        if n.mass == 0 or (n.point is not None and np.all(n.point[0] == point)):
            return np.zeros(2)
        dist = np.linalg.norm(n.mass_center - point) + eps
        if len(n.children) == 0 or n.size / dist < theta:
            qij = 1.0 / (1.0 + dist ** 2)
            force = qij * qij * (point - n.mass_center) * n.mass
            return force
        else:
            return sum([recurse(c) for c in n.children])
    return recurse(node)

# Paso 4: Descenso del gradiente
def _compute_low_dim_affinities(Y):
    distances = squareform(pdist(Y, 'euclidean')) ** 2
    inv_distances = 1 / (1 + distances)
    np.fill_diagonal(inv_distances, 0)
    Q = inv_distances / np.sum(inv_distances)
    return Q, inv_distances

def gradient_descent_bh(Y, P, learning_rate=200.0, n_iter=1000, theta=0.5):
    n, dim = Y.shape
    dY = np.zeros((n, dim))
    iY = np.zeros((n, dim))
    gains = np.ones((n, dim))

    for iter in range(n_iter):
        Q, num = _compute_low_dim_affinities(Y)
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (dim, 1)).T * (Y[i, :] - Y), axis=0)

        tree = build_quadtree(Y)
        for i in range(n):
            force = compute_repulsion_force(tree, Y[i], theta)
            dY[i] += force

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        iY = 0.5 * iY - learning_rate * (gains * dY)
        Y += iY
        Y -= np.mean(Y, axis=0)

        if iter % 100 == 0:
            print(f"Iter {iter}")
    return Y