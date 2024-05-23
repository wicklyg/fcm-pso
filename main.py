import numpy as np
import pandas as pd
from src.fcm import FCM
from src.fpso import PSO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    # Load data from Excel file
    data = pd.read_excel("D:/fcm-pso/data/Data IKR New.xlsx")
    df = data.drop(['Kab/kota', 'Provinsi'], axis=1)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)
    
    # Convert scaled and PCA data to DataFrame for saving
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    
    # Save standardized and PCA data to Excel
    with pd.ExcelWriter("D:/fcm-pso/data/Processed_Data.xlsx") as writer:
        scaled_df.to_excel(writer, sheet_name="Standardized Data", index=False)
        pca_df.to_excel(writer, sheet_name="PCA Data", index=False)
    
    # Parameters for FCM
    n_cluster = 3
    m = 2
    max_iter = 100

    # Run FCM
    fcm = FCM(n_cluster=n_cluster, m=m, max_iter=max_iter)
    fcm.fit(pca_data)
    fcm_clusters = fcm.predict()
    
    print("FCM Clusters:", fcm_clusters)
    print("FCM Centroids:\n", fcm.center)
    print("FCM Objective Function Value:\n", fcm.obj_func)

    # Parameters for PSO
    n_particles = 10
    max_iter_pso = 10
    c1 = 1.5
    c2 = 1.5
    w = 0.5

    # Run PSO
    pso = PSO(n_cluster=n_cluster, n_particle=n_particles, data=pca_data, max_iter=max_iter_pso)
    pso.run()

    # Show FCM results
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=fcm_clusters, cmap='viridis')
    plt.scatter(fcm.center[:, 0], fcm.center[:, 1], color='red', marker='x')
    plt.title('FCM Clustering')
    plt.show()

if __name__ == "__main__":
    main()