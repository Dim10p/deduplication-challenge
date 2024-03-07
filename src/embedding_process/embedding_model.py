"""
This file contains the overall embedding model script which is applied on the formatted data.
"""

# import packages/libraries:
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# activate tqdm pandas:
tqdm.pandas()

# import constants and functions:
from src.utilities.constants import Paths, DataDimensions


def create_overall_embedding(language_embedding_model: SentenceTransformer) -> None:
    """
    The goal of this function is to create the embedding and store it.
    The strategy is to only compute embeddings for each unique (translated) description with its title (~59000).

    :param language_embedding_model: mpnet, minilm or distilbert
    """

    # read in formatted data:
    df = pd.read_parquet(Paths.INTERMEDIATE_DATA_PATH)

    # drop duplicate descriptions, no point in computing those multiple times (but keep the hash to bring them back):
    map_description_to_hash = (
        df.filter([DataDimensions.TITLE_AND_DESCRIPTION, DataDimensions.TITLE_AND_DESCRIPTION_HASH])
        .drop_duplicates(keep='first')
        .sort_values(DataDimensions.TITLE_AND_DESCRIPTION_HASH)
        .reset_index(drop=True)
    )

    # force title and description hash to int data type:
    map_description_to_hash[DataDimensions.TITLE_AND_DESCRIPTION_HASH] = map_description_to_hash[
        DataDimensions.TITLE_AND_DESCRIPTION_HASH].astype(int)

    # create a corpus of all descriptions (list):
    description_corpus = list(map_description_to_hash[DataDimensions.TITLE_AND_DESCRIPTION].unique())

    # start embedding (locally, this takes 2-3 hours on the whole dataset):
    corpus_embedding = (language_embedding_model.encode(
        description_corpus,
        convert_to_tensor=True,
        show_progress_bar=True))

    # store the embedding output in order to be used later:
    embedding_path = Paths.get_embedding_path(language_embedding_model.get_name())
    torch.save(corpus_embedding, f=embedding_path)
    print(f"Embedding saved to {embedding_path}.")

    # store the description to hash correspondence to parquet:
    map_description_to_hash.to_parquet(Paths.MAP_DESCRIPTION_TO_HASH)
    print(f"map_description_to_hash saved to {Paths.MAP_DESCRIPTION_TO_HASH}.")
