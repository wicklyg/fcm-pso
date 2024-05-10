import numpy as np

from fcm import FCM
from utils import fcm_obj_func

class particle:
    def __init__(self,
                 n_cluster: int,
                 data,
                 w: float = 0.4,
                 c1: float = 2.0,
                 c2: float = 2.0):
        
        global FCM
        fcm = FCM(n_cluster=n_cluster)
        fcm.fit(data)

        self.membership = fcm._init_membership_matrix(data)
        
        # initialize params
        self._w = w
        self._c1 = c1
        self._c2 = c2
        self.r1 = np.random.uniform()
        self.r2 = np.random.uniform()
        self.r3 = np.random.normal(0, 1)
        self.r4 = np.random.noncentral_f(0, 1)
        self.n_cluster = n_cluster

        #initialize particle
        self.centroids = fcm._calc_centroids(data, self.membership)
        self.pbest_position = self.membership.copy()
        self.pworst_position = self.membership.copy()
        self.best_fitness = fcm_obj_func(data, self.membership, self.centroids, 2)
        self.worst_fitness = fcm_obj_func(data, self.membership, self.centroids, 2)

        self.velocity = np.zeros_like(self.membership)


    def update(self, gbest_position, gworst_position, data):
        self._update_velocity(gbest_position, gworst_position)
        self._update_membership(data)

    def _update_velocity(self, gbest_postition, gworst_position):

        v_old = self._w * self.velocity

        cognitive_component = self._c1 * self.r1 * (self.pbest_position - self.membership)  +\
                            self.r3 * (self.pworst_position - self.membership)

        social_component = self._c2 * self.r2 * (gbest_position - self.membership) +\
                        self.r4 * (gworst_position - self.membership)

        self.velocity = v_old + cognitive_component + social_component

# update the solution position and normalize membership
    def _update_membership(self, data):
        n = data.shape[0]
        new_membership = self.membership + self.velocity

        for k in range(n):
            for i in range(self.n_cluster):
                if new_membership[k, i] <= 0:
                    new_membership[k, i] = 0

        for k in range(n):
            summation = np.sum(new_membership[k, :])
            self.membership[k] = new_membership[k] / summation

        self.centroids = fcm._calc_centroids(data, self.membership)
        self.membership = fcm._update_membership_matrix(data, self.centroids, self.membership)
        new_fitness = fcm_obj_func(data, self.membership, self.centroids, self.n_cluster, 2)

        if new_fitness < self.best_fitness:
            self.best_fitness = new_fitness
            self.pbest_position = self.membership.copy()
        if new_fitness > self.worst_fitness:
            self.worst_fitness = new_fitness
            self.pworst_position = self.membership.copy()

        return self


    def _predict(self):
        cluster = self._assign_cluster()
        return cluster


    def _assign_cluster(self):
        cluster = np.argmax(self.membership, axis=1)
        return cluster