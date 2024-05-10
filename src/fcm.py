import numpy as np

def init_membership_matrix(data: np.ndarray, n_cluster: int):
    n_data = data.shape[0]
    membership_matrix = np.random.rand(n_data, n_cluster)
    membership_matrix /= np.sum(membership_matrix, axis=1)[:, np.newaxis]
    return membership_matrix

def calculate_obj_function(data, centroids, membership_matrix, fuzzines):
    num_clusters = centroids.shape[0]
    num_data = data.shape[0]
    objective_function = 0
    for i in range(num_data):
        for j in range(num_clusters):
            distance_to_cluster_j = np.linalg.norm(data[i] - centroids[j])
            objective_function += (membership_matrix[i, j]**fuzzines)*(distance_to_cluster_j**2)

    return objective_function


class FCM:
    def __init__(self, 
                n_cluster: int,
                max_iter: int = 10,
                tolerance: float = 1e-4,
                fuzziness: float = 2.0,
                seed: int = None):
        
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fuzziness = fuzziness
        self.seed = seed
        self.membership_matrix = None
        self.centroids = None
        self.fcm_objective_function = None

    def fit(self, data: np.ndarray):
        self.membership_matrix = self._init_membership_matrix(data=data)
        self.objective_function = 0

        for iteration in range(self.max_iter):
            new_centroids = self._calc_centroids(data, self.membership_matrix)
            new_membership_matrix = self._update_membership_matrix(data, new_centroids, self.membership_matrix)
            
            self.membership_matrix = new_membership_matrix
            self.centroids = new_centroids
            
            obj_function = calculate_obj_function(data, self.centroids, self.membership_matrix, self.fuzziness)
            diff = obj_function - self.objective_function
            self.objective_function = obj_function

            print(f"Iteration {iteration + 1}/{self.max_iter} - Objective Function: {self.objective_function}")
            print("Membership Matrix:")
            print(self.membership_matrix)

            if np.abs(diff) < self.tolerance:
               break
        return self

    def predict(self, data: np.ndarray):
        return self._assign_cluster(self.membership_matrix)

    def _init_membership_matrix(self, data: np.ndarray):
        n_data = data.shape[0]
        membership_matrix = np.random.rand(n_data, self.n_cluster)
        membership_matrix /= np.sum(membership_matrix, axis=1)[:, np.newaxis]
        return membership_matrix
    
    def _calc_centroids(self, data: np.ndarray, membership_matrix: np.ndarray):
        centroids = np.zeros((self.n_cluster, data.shape[1]))
        for i in range(self.n_cluster):
            numerator = np.sum((membership_matrix[:, i]**self.fuzziness)[:,np.newaxis]* data, axis = 0)
            denominator = np.sum(membership_matrix[:, i]**self.fuzziness)
            centroids[i, :] = numerator/denominator
        return centroids

    def _update_membership_matrix(self, data: np.ndarray, centroids: np.ndarray, membership_matrix):
        n_data = data.shape[0]
        update_membership_matrix = membership_matrix

        for i in range(n_data):
            total_distance = np.sum([np.linalg.norm(data[i] - centroids[k]) for k in range(self.n_cluster)])
            for j in range(self.n_cluster):
                distance_to_cluster_j = np.linalg.norm(data[i] - centroids[j])
                update_membership_matrix [i, j] = 1 / np.sum((distance_to_cluster_j / (total_distance)**(2 / (self.fuzziness-1))))

        update_membership_matrix /= np.sum(update_membership_matrix, axis=1)[:, np.newaxis]

        return update_membership_matrix
    
    def _assign_cluster(self, membership_matrix: np.ndarray):
        return np.argmax(membership_matrix, axis=1 )

if __name__ == "__main__":
    pass