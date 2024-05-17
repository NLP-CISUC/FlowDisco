from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
import pickle


def clustering_kmeans(vectors, n_clusters, nomeFichPickle):
    """Receive vectors and do the clustering in KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=1000, random_state=2)
    kmeans.fit(vectors)
    labels_kmeans = kmeans.labels_
    centers_kmeans = kmeans.cluster_centers_
    print(
        "\nInternal Evaluation:\nSilhouette Score: ",
        metrics.silhouette_score(vectors, labels_kmeans),
    )
    print(
        "Davies-Bouldin Index (DBI): ",
        metrics.davies_bouldin_score(vectors, labels_kmeans),
    )

    # Save the model to a file
    with open(nomeFichPickle, 'wb') as f:
      pickle.dump(kmeans, f)

    print("labels_kmeans", labels_kmeans)
    return labels_kmeans, centers_kmeans


def clustering_dbscan(vectors, eps, minsamples):
    """Receive vectors and do the clustering in DBSCAN (-1 cluster represents outliers)."""
    dbscan = DBSCAN(eps=eps, min_samples=minsamples)
    dbscan.fit(vectors)
    labels_dbscan = dbscan.labels_
    print(labels_dbscan)

    print(
        "\nInternal Evaluation:\nSilhouette Score: ",
        metrics.silhouette_score(vectors, labels_dbscan),
    )
    print(
        "Davies-Bouldin Index (DBI): ",
        metrics.davies_bouldin_score(vectors, labels_dbscan),
    )

    return labels_dbscan
