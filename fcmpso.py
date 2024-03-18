import numpy as np

from particle import Particle
from fcm import init_membership_matrix

class FPSO:
    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 max_iter: int = 100,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.print_debug = print_debug
        self.gbest_score = np.inf
        self.gbest_membership_matrix = None
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            particle = Particle(data = self.data, n_cluster=self.n_cluster)
            if particle.best_score < self.gbest_score:
                self.gbest_membership_matrix = particle.membership_matrix.copy()
                self.gbest_score = particle.best_score
            self.particles.append(particle)
            self.gbest_score = min(particle.)

    
    def run(self):
        print('Initial global best score', self.gbest_score)

        history = []
        for i in range(self.max_iter):
            for particle in self.particles:
                particle = Particle(data = self.data, n_cluster=self.n_cluster)
                particle.update(self.gbest_membership_matrix)