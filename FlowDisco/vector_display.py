import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def create_pca(normalized_df, vectors, n_clusters):
    y_true = normalized_df["trueLabel"]

    scaler = StandardScaler()
    x = scaler.fit_transform(vectors)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(x)

    # sns settings
    sns.set(rc={"figure.figsize": (13, 9)})

    # colors
    palette = sns.hls_palette(n_clusters, l=0.4, s=0.9)

    # plot
    sns.scatterplot(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        hue=y_true,
        legend="full",
        palette=palette,
    )
    plt.title("PCA with Kmeans Labels")
    # plt.savefig("improved_cluster_tsne.png")
    plt.show()


def create_tsne(normalized_df, vectors, n_clusters):
    y_true = normalized_df["trueLabel"]

    scaler = StandardScaler()
    x = scaler.fit_transform(vectors)

    # Changed perplexity from 100 to 50 per FAQ
    tsne = TSNE(verbose=1, perplexity=50, learning_rate="auto", init="random")
    x_embedded = tsne.fit_transform(x)

    # sns settings
    sns.set(rc={"figure.figsize": (13, 9)})

    # colors
    palette = sns.hls_palette(n_clusters, l=0.4, s=0.9)

    # plot
    sns.scatterplot(
        x=x_embedded[:, 0],
        y=x_embedded[:, 1],
        hue=y_true,
        legend="full",
        palette=palette,
    )
    plt.title("t-SNE with Kmeans Labels")
    # plt.savefig("improved_cluster_tsne.png")
    plt.show()
