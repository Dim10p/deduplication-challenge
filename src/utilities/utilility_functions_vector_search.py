"""
This file contains the function for semantic match row by row.
"""

# import packages/libraries:
from typing import Dict
import faiss
import numpy as np
from torch import Tensor


def find_semantic_match_for_each_row(
        corpus_embedding_description_hash: Tensor,
        index: faiss.IndexIDMap,
        k_nearest_neighbours: int = 20) -> Dict[int, int]:
    """
    Finds the k_nearest_neighbours matches of description_hash in corpus_embedding via index.

    :param Tensor corpus_embedding_description_hash: one tensor from the corpus embedding
    :param faiss.IndexIDMap index: populated index
    :param int k_nearest_neighbours: number of matches to be returned
    """

    np_array = np.array([corpus_embedding_description_hash])
    l2_distance, index_location = index.search(np_array, k=k_nearest_neighbours)
    result_dict = dict(zip(
        [index_id for index_id in index_location.tolist()[0]],
        l2_distance.tolist()[0]))

    return result_dict
