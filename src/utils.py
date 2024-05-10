import numpy as np

def fcm_obj_func(data, membership, centroids, fuzziness):
    n_data = data.shape[0]
    n_cluster = membership.shape[1]
    J = 0.0
    for k in range(n_data):
        for i in range(n_cluster):
            distance[k, i] = np.linalg.norm(data[k] - centroids[i])
            J += (membership[k, i] ** fuzziness) * (distance[k, i] ** 2)
    
    return J