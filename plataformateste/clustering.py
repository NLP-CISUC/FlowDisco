from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans


def clustering_kmeans(vectors, n_clusters):
    """Receive vectors and do the clustering in KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500)
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
