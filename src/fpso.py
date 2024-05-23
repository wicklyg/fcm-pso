import numpy as np
import matplotlib.pyplot as plt
from src.fcm import FCM
from src.particle import Single_Particle, fcm_obj_func

class PSO:
    def __init__(self, n_cluster: int, n_particle: int, data, hybrid: bool = True, max_iter: int = 10, print_debug: int = 1):
        global fcm
        fcm = FCM(n_cluster=n_cluster, m=2)
        self.n_cluster = n_cluster
        self.n_particle = n_particle
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.print_debug = print_debug
        self.gbest_fitness = np.inf
        self.gworst_fitness = 0.0
        self.gbest_position = None
        self.gworst_position = None
        self.final_centroid = None
        self.cluster = None
        self._init_particle()

    def _init_particle(self):
        for i in range(self.n_particle):
            particle = Single_Particle(self.n_cluster, self.data)
            if particle.best_fitness < self.gbest_fitness:
                self.gbest_position = particle.pbest_position.copy()
                self.gbest_fitness = particle.best_fitness
            if particle.worst_fitness > self.gworst_fitness:
                self.gworst_position = particle.pworst_position.copy()
                self.gworst_fitness = particle.worst_fitness
            self.particles.append(particle)

    def run(self):
        print('Initial global best fitness:', self.gbest_fitness)
        print('Initial global worst fitness:', self.gworst_fitness)
        history = []
        for i in range(self.max_iter):
            for particle in self.particles:
                if particle.best_fitness < self.gbest_fitness:
                    self.gbest_position = particle.pbest_position.copy()
                    self.gbest_fitness = particle.best_fitness
                if particle.worst_fitness > self.gworst_fitness:
                    self.gworst_position = particle.pworst_position.copy()
                    self.gworst_fitness = particle.worst_fitness
                particle.update(self.gbest_position, self.gworst_position, self.data)
                self.final_centroid = fcm._cal_center(self.data, self.gbest_position)
                self.gbest_position = fcm._update_membership(self.data, self.final_centroid)
            for _ in range(10):
                self.final_centroid = fcm._cal_center(self.data, self.gbest_position)
                self.gbest_position = fcm._update_membership(self.data, self.final_centroid)
                self.gbest_fitness = fcm_obj_func(self.data, self.final_centroid, self.gbest_position, self.n_cluster, 2)
            history.append(self.gbest_fitness)
            if i % self.print_debug == 0:
                print(f'Iteration {i+1}/{self.max_iter} current gbest fitness {self.gbest_fitness:.13f} gworst fitness {self.gworst_fitness:.13f}')
        self.final_centroid = fcm._cal_center(self.data, self.gbest_position)
        self.cluster = np.argmax(self.gbest_position, axis=1)
        print(f'Finish with gbest score {self.gbest_fitness:.18f}')
        print('Final membership:', self.gbest_position)
        print('Final centroid', self.final_centroid)
        print('cluster:', self.cluster)
        plt.plot(history)
        plt.title('convergence curve')
        plt.ylabel('fitness')
        plt.ylim(600, 9000)
        plt.xlabel('iteration')
        plt.show()
        return history