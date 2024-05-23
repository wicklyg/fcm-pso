import numpy as np

def _obj_func(data, center, u, c, m):
    n = data.shape[0]
    J = 0.0
    distance = np.zeros((n, c))
    for k in range(n):
        for i in range(c):
            distance[k, i] = np.linalg.norm(data[k] - center[i])
            J += (u[k, i] ** m) * (distance[k, i] ** 2)
    return J

class FCM:
    def __init__(self, n_cluster: int, m, max_iter: int = 15, sigma: float = 1e-3):
        self.n_cluster = n_cluster
        self.m = m
        self.max_iter = max_iter
        self.sigma = sigma
        self.center = None
        self.U = None
        self.obj_func = None
        self.cluster = None

    def _init_membership(self, data):
        n = data.shape[0]
        u = np.zeros((n, self.n_cluster))
        for k in range(n):
            random_list = np.random.rand(self.n_cluster)
            summation = np.sum(random_list)
            u[k] = random_list / summation
        self.U = u
        return u

    def _cal_center(self, data, u):
        n, p = data.shape
        center = np.zeros((self.n_cluster, p))
        for i in range(self.n_cluster):
            sample_sum = np.zeros(p)
            member_sum = 0.0
            for k in range(n):
                temp1 = (u[k, i] ** self.m)
                temp2 = temp1 * data[k]
                member_sum += temp1
                sample_sum += temp2
            center[i] = sample_sum / member_sum
        return center

    def _update_membership(self, data, center):
        n = data.shape[0]
        t = -(2 / (self.m - 1))
        u = np.zeros((n, self.n_cluster))
        distance = np.zeros((n, self.n_cluster))
        for k in range(n):
            for i in range(self.n_cluster):
                distance[k][i] = np.linalg.norm(data[k] - center[i])
        for k in range(n):
            for i in range(self.n_cluster):
                u[k, i] = (distance[k, i] ** t) / np.sum(distance[k, :] ** t)
        return u

    def _assign_cluster(self):
        self.cluster = np.argmax(self.U, axis=1)
        return self.cluster

    def fit(self, data):
        self.U = self._init_membership(data)
        count = 0
        while count <= self.max_iter:
            self.center = self._cal_center(data, self.U)
            self.U = self._update_membership(data, self.center)
            count += 1
        self.obj_func = _obj_func(data, self.center, self.U, self.n_cluster, self.m)
        return self

    def predict(self):
        return self._assign_cluster()