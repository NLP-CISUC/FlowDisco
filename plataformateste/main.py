import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from plataformateste import graphs
from plataformateste.platform import run_test

DATA_FOLDER = Path(__file__).parent.parent / "data"
RESULTS_FOLDER = Path(__file__).parent.parent / "results"
MODEL_CACHE_FOLDER = Path(__file__).parent.parent / "models"


@click.command()
@click.option("--data_filename")
@click.option("--package", default="en_core_web_md")
@click.option("--representation", default="sentenceTransformer")
@click.option("--labels-type", default="bigrams")
@click.option("--n-clusters", type=int)
def main(
    data_filename: str,
    package: str,
    representation: str,
    labels_type: str,
    n_clusters: int
):
    save_file_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(DATA_FOLDER / data_filename, on_bad_lines="skip", sep=";")
 
    graph = run_test(
        df, package, representation, labels_type, n_clusters, MODEL_CACHE_FOLDER
    )

    graphs.save(graph, RESULTS_FOLDER / f"{save_file_id}_{representation}_{labels_type}.dot")

    with open(RESULTS_FOLDER / f"{save_file_id}_config.json", "w") as f:
        json.dump(
            {
                "data_filename": data_filename,
                "package": package,
                "representation": representation,
                "labels_type": labels_type,
                "n_clusters": n_clusters
            },
            f
        )


if __name__ == "__main__":
    main()
