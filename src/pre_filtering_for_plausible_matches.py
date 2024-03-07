"""
This file contains the pre-filtering wrapper function.
"""

# import packages/libraries:
from src.embedding_process.embedding_model import create_overall_embedding
from src.embedding_process.index_search import use_embedding
from src.embedding_process.reassemble_dataframe import merge_results_to_dataframe

# import SentenceTransformer object:
from sentence_transformers import SentenceTransformer


def pre_filtering_for_plausible_matches(
        language_embedding_model: SentenceTransformer,
        n_matches: int,
        pre_filtering_l2_threshold: float):
    """
    Wrapper around embedding model. Three steps process:
    - Create the overall embedding
    - Search for the closest matches
    - Put everything back together

    :param SentenceTransformer language_embedding_model: language embedding model
    :param int n_matches: number of the closest matches
    :param float pre_filtering_l2_threshold: L2 distance threshold
    """

    create_overall_embedding(language_embedding_model)

    use_embedding(language_embedding_model, n_matches)

    merge_results_to_dataframe(language_embedding_model,
                               pre_filtering_l2_threshold)
