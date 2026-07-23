import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

def elbow_silhouette_method(df):


    inertia = []
    k_range = [2,3,4,5,6,7,8]
    sil_score = []

    for k in k_range:
        km = KMeans(n_clusters=k,max_iter=50,random_state=42)
        km.fit(df)
        inertia.append(km.inertia_)
        sil_score.append(silhouette_score(df,km.labels_))

    print(f"inertia : {inertia} of k_range : {k_range}\n")


    return k_range,inertia,sil_score
    

    # # Plot elbow curve safely
    # plt.figure(figsize=(6,4))
    # plt.plot(k_range, inertia, marker='o')
    # plt.xlabel("Number of clusters (k)")
    # plt.ylabel("Inertia")
    # plt.title("Elbow Method")

    # try:
    #     plt.show()   # works if interactive backend is available
    # except Exception:
    #     # fallback: save instead of show
    #     plt.savefig("elbow_method.png")
    #     print("Plot saved as elbow_method.png (backend issue prevented plt.show).")


def k_means(df,clusters):
    #final model with k=cluster

    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels =kmeans.fit_predict(df)
    df["Cluster"] = kmeans.labels_
    return kmeans, labels

def agglo_clustering(df):

    hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels = hierarchical.fit_predict(df)
    return hierarchical, labels

def compute_linkage_matrix(df):
 
    # linkage returns a matrix containing clustering information (which nodes merged, distances, etc.)
 
    return linkage(df, method="ward")