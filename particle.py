import numpy as np

from fcm import FCM, calculate_obj_function, init_membership_matrix

def calc_centroids(data: np.ndarray, 
                   n_cluster: int, 
                   membership_matrix: np.ndarray, 
                   fuzziness: float):
        centroids = np.zeros((n_cluster, data.shape[1]))
        for i in range(n_cluster):
            numerator = np.sum((membership_matrix[:, i]**fuzziness)[:,np.newaxis]* data, axis = 0)
            denominator = np.sum(membership_matrix[:, i]**fuzziness)
            centroids[i, :] = numerator/denominator
        return centroids

class Particle:
    def __init__(self, 
                 n_cluster, 
                 data, 
                 fuzziness: float = 2.0, 
                 w: float = 0.9, 
                 c1: float = 0.5, 
                 c2: float = 0.3):
        self.n_cluster = n_cluster
        self.fuzziness = fuzziness
        self.membership_matrix = init_membership_matrix(data=data, n_cluster=self.n_cluster)
        self.centroids = calc_centroids(data, self.n_cluster, self.membership_matrix, self.fuzziness)
        self.best_position = self.membership_matrix.copy()
        self.best_score = calculate_obj_function(data, self.centroids, self.membership_matrix, self.fuzziness)
        self.velocity = np.zeros_like(self.membership_matrix)
        self.data = data
        self._w = w
        self._c1 = c1
        self._c2 = c2

    def update(self, gbest_postition: np.ndarray, data: np.ndarray):
        self._update_velocity(gbest_postition)
        return self._update_membership_matrix(data, self.centroids, self.membership_matrix)
        
    def _update_velocity(self, gbest_postition: np.ndarray):
        v_old = self._w * self.velocity
        cognitive_component = self._c1 * np.random.random() * (self.best_position - self.membership_matrix)
        social_component = self._c2 * np.random.random() * (gbest_postition - self.membership_matrix)
        self.velocity = v_old + cognitive_component + social_component
        
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
        
    def _predict(self, data: np.ndarray) -> np.ndarray:
        cluster = self._assign_cluster(self.membership_matrix)
        return cluster
        
    def _assign_cluster(self, membership_matrix: np.ndarray):
        return np.argmax(membership_matrix, axis=1)
        

if __name__ == "__main__":
    data = np.random.rand(30, 2)
    n_cluster = 2
    particle = Particle(n_cluster=n_cluster, data=data)
    max_iterations = 50

    for iteration in range(max_iterations):
        gbest_position = particle.best_position
        gbest = particle.update(gbest_position, data)

        print(f"Iteration {iteration + 1}/{max_iterations} - gbest: {gbest}")