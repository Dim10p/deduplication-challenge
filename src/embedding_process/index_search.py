"""
This file contains the index search script which is applied on the embedded data.
"""

# import packages/libraries:
import pickle
from typing import Dict

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

# import tqdm pandas:
tqdm.pandas()

# import function for semantic match and constants:
from src.utilities.utilility_functions_vector_search import find_semantic_match_for_each_row
from src.utilities.constants import Paths, DataDimensions


def perform_index_search(corpus_embedding: Tensor,
                         map_description_to_hash: pd.DataFrame,
                         language_embedding_model_name: str,
                         n_matches: int) -> Dict[int, Dict[int, float]]:
    """
    Finds matches

    :param corpus_embedding:
    :param map_description_to_hash: Dataframe
    :param language_embedding_model_name: the name of the
    :param n_matches: number of matches for each row
    """

    # convert the corpus to numpy array (cpu before, to work in AWS as well):
    corpus_embedding = corpus_embedding.cpu().numpy()

    # convert it to float32 data types:
    corpus_embedding = np.float32(corpus_embedding)

    # create the index object:
    index = faiss.IndexFlatL2(corpus_embedding.shape[1])
    index = faiss.IndexIDMap(index)

    # add based on ids:
    index.add_with_ids(corpus_embedding, map_description_to_hash[DataDimensions.TITLE_AND_DESCRIPTION_HASH].values)

    print(f"Number of vectors in the Faiss index: {index.ntotal}")

    # initialize an empty dictionary to fill:
    results = dict()
    pbar = tqdm(map_description_to_hash[DataDimensions.TITLE_AND_DESCRIPTION_HASH].unique())
    pbar.set_description(f"faiss search for matches in {language_embedding_model_name} embedding")
    for description_hash in pbar:
        # fill in each description hash with its k nearest neighbours:
        results[int(description_hash)] = find_semantic_match_for_each_row(corpus_embedding[description_hash],
                                                                          index,
                                                                          k_nearest_neighbours=n_matches)

    # return the dictionary of dictionaries:
    return results


def use_embedding(language_embedding_model: SentenceTransformer, n_matches: int = 100):
    """
    Here we use the embedding to find the k nearest neighbours of each unique cleaned description. We store the
    result in a dictionary of dictionaries with the following structure:
    {
     hash_of_the_cleaned_description : {hash_of_another_cleaned_description: l2_distance, ....}
    }

    so, something like this:
    {
        1: {12: 0.1456, 456:0.123,...},
        2: {8: 0.654,...}
    }

    :param SentenceTransformer language_embedding_model: minilm or distilbert
    """

    # read in the data and the embedding corpus:
    map_description_to_hash = pd.read_parquet(Paths.MAP_DESCRIPTION_TO_HASH)
    corpus_embedding = torch.load(f=Paths.get_embedding_path(language_embedding_model.get_name()))

    # perform index search to all ids:
    results = perform_index_search(
        corpus_embedding,
        map_description_to_hash,
        language_embedding_model.get_name(),
        n_matches=n_matches)

    # store the results:
    # https://www.digitalocean.com/community/tutorials/python-pickle-example
    with open(Paths.get_path_for_map_description_to_hash_with_matches(language_embedding_model.get_name()), 'wb') as f:
        # pickle the 'data' dictionary using the highest protocol available:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
