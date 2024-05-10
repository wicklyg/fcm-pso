class FPSO_Clustering:
    def __init__(self, n_clusters=3, population_size=30, c1=2, c2=2, w=0.5, max_iter=100, constant_K=1):
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.max_iter = max_iter
        self.pbest = None
        self.gbest = None
        self.constant_K = constant_K

    def initialize_population(self, n_samples):
        return np.random.rand(self.population_size, n_samples, self.n_clusters)

    def update_velocity(self, V, X, pbest, gbest):
        r1 = np.random.rand(self.population_size, 1, self.n_clusters)
        r2 = np.random.rand(self.population_size, 1, self.n_clusters)
        return self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)

    def update_position(self, X, V):
        return X + V

    def normalize_position(self, X):
        X_normalized = np.maximum(X, 0)
        row_sums = X_normalized.sum(axis=1)
        return X_normalized / row_sums[:, np.newaxis]

    def fitness_function(self, X, data):
        cluster_centers = X.reshape(-1, data.shape[1])
        distances = pairwise_distances(data, cluster_centers)
        memberships = 1 / (1 + distances**2)
        J_U_Z = np.sum(memberships ** 2)
        return self.constant_K / J_U_Z

    def fit(self, data):
        n_samples, n_features = data.shape
        X = self.initialize_population(n_samples)
        V = np.random.uniform(-1, 1, size=(self.population_size, n_samples, self.n_clusters))

        for _ in range(self.max_iter):
            fitness_values = np.array([self.fitness_function(x, data) for x in X])

            if self.gbest is None or np.min(fitness_values) < self.fitness_function(self.gbest, data):
                self.gbest = X[np.argmin(fitness_values)]

            if self.pbest is None:
                self.pbest = X.copy()

            improved = fitness_values < self.fitness_function(self.pbest, data)
            self.pbest[improved] = X[improved]

            V = self.update_velocity(V, X, self.pbest, self.gbest)
            X = self.update_position(X, V)
            X = self.normalize_position(X)
            self.gbest = self.normalize_position(self.gbest)

        return self.gbest, X