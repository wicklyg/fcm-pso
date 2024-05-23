import numpy as np

def calculate_validity_indices(gbest_position, final_centroid, data, n_cluster):
    min_dist = np.min([np.linalg.norm(final_centroid[i] - final_centroid[j]) for i in range(n_cluster) for j in range(i + 1, n_cluster)])
    xb = np.sum(gbest_position**2 * np.linalg.norm(data[:, np.newaxis] - final_centroid, axis=2)**2) / (data.shape[0] * min_dist**2)
    pc = np.sum(gbest_position**2) / data.shape[0]
    vi = np.sum(gbest_position * np.linalg.norm(data[:, np.newaxis] - final_centroid, axis=2)**2, axis=0) / np.sum(gbest_position, axis=0)
    pi = np.sum(vi) / min_dist
    s = min_dist
    return pc, xb, pi, s