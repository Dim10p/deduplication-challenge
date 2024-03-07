"""
This file contains the merge script which is applied between the embedding data and the original data.
"""

# import packages/libraries:
import os
import pickle
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

# import data processing functions, and constants:
from src.utilities.constants import Paths, DataDimensions, OutputDataDims
from src.utilities.utility_functions_data_processing import (explode_into_final_format,
                                                             assign_semantic_or_partial_matches_to_id_via_description_hashes,
                                                             create_dict_of_semantic_and_temporal_vs_partial_matches,
                                                             add_semantic_and_temporal_matches_columns,
                                                             keep_best_match_per_id1_id2_combination)

# activate tqdm pandas:
tqdm.pandas()

from sentence_transformers import SentenceTransformer


def compute_merge_results_to_dataframe(
        challenge_data: pd.DataFrame,
        full_matches: pd.DataFrame,
        match_database: Dict[int, Dict[int, float]],
        prefiltering_l2_threshold: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    In this function, I merge back the original data. Remember that many IDs have the same cleaned descriptions.
    So those will all have each other as semantic duplicates (not perfect duplicates, this we compute somewhere else).
    moreover they all have the same matches computed in the previous functions. Consider this example:
    ID description     description_clean  description_hash  identical_cleaned_descriptions_ids
    1  "A great job*"  "a great job"      1                 [1, 2]
    2  "A great job"   "a great job"      1                 [1, 2]
    3  "An great job"  "an great job"     2                 [3]

    The cleaned descriptions are identical for the first two rows, so they get the same description hash. I also added
    a column that contains a list where for each ID you can see which other IDs have the same cleaned description.
    In the embedding they only appear once, so the embedding uses the following data:

    description_hash   description_clean
    1                  "a great job"
    2                  "an great job"

    Those will be quite close in l2 distance, so likely they will be matched as semantic duplicates.

    Semantic duplicates are then the sum of identical_cleaned_descriptions_ids and the results of the embedding match
    subject to threshold of how large a l2 distance one would consider.


    :param challenge_data:
    :param match_database:
    :param prefiltering_l2_threshold:
    :return:
    """

    semantic_and_temporal_matches = create_dict_of_semantic_and_temporal_vs_partial_matches(
        match_database=match_database,
        lower_threshold=prefiltering_l2_threshold)

    # a dictionary mapping hashes of cleaned descriptions to (multiple) ids
    description_hash_to_ids = challenge_data.groupby(DataDimensions.TITLE_AND_DESCRIPTION_HASH)[DataDimensions.ID].agg(
        list).to_dict()

    challenge_data[DataDimensions.IDENTICAL_CLEANED_DESCRIPTIONS_IDS] = challenge_data[
        DataDimensions.TITLE_AND_DESCRIPTION_HASH].map(
        description_hash_to_ids)

    tqdm.pandas(desc="Assigning semantic and temporal matches")
    challenge_data[DataDimensions.SEMANTIC_AND_TEMPORAL_MATCHES] = challenge_data.progress_apply(
        lambda row: assign_semantic_or_partial_matches_to_id_via_description_hashes(
            identical_cleaned_descriptions_ids=row[DataDimensions.IDENTICAL_CLEANED_DESCRIPTIONS_IDS],
            description_hash=row[DataDimensions.TITLE_AND_DESCRIPTION_HASH],
            description_hash_to_ids=description_hash_to_ids,
            semantic_or_partial_matches=semantic_and_temporal_matches),
        axis=1
    )

    del description_hash_to_ids, semantic_and_temporal_matches

    challenge_data = add_semantic_and_temporal_matches_columns(challenge_data)

    semantic_matches = pd.DataFrame(challenge_data, copy=True).filter([DataDimensions.ID,
                                                                       DataDimensions.SEMANTIC_MATCHES])

    temporal_matches = pd.DataFrame(challenge_data, copy=True).filter([DataDimensions.ID,
                                                                       DataDimensions.TEMPORAL_MATCHES])

    semantic_matches_final = explode_into_final_format(df=semantic_matches,
                                                       match_type=DataDimensions.SEMANTIC_MATCHES)

    temporal_matches_matches_final = explode_into_final_format(df=temporal_matches,
                                                               match_type=DataDimensions.TEMPORAL_MATCHES)

    df_concat = pd.concat(
        [
            full_matches,
            semantic_matches_final,
            temporal_matches_matches_final
        ]
    )

    df_concat[OutputDataDims.ID1] = df_concat[OutputDataDims.ID1].astype(int)
    df_concat[OutputDataDims.ID2] = df_concat[OutputDataDims.ID2].astype(int)

    # keep the most granular for each
    df_concat = keep_best_match_per_id1_id2_combination(df_concat)

    # make sure to only export cases where ID1 < ID2:
    df_concat = df_concat[df_concat[OutputDataDims.ID1] < df_concat[OutputDataDims.ID2]]

    return df_concat, challenge_data


def merge_results_to_dataframe(
        language_embedding_model: SentenceTransformer,
        prefiltering_l2_threshold: float):
    """
    This function is just a wrapper so I/O and computation is decoupled

    :param language_embedding_model:
    :param prefiltering_l2_threshold:
    :return:
    """

    # read the entire dataset:
    formatted_data = pd.read_parquet(Paths.INTERMEDIATE_DATA_PATH)

    # get the results
    with open(Paths.get_path_for_map_description_to_hash_with_matches(language_embedding_model.get_name()), 'rb') as f:
        match_database = pickle.load(f)

    # remember that full matches contains some full matches:
    full_matches = pd.read_parquet(Paths.FULL_DUPLICATES_PATH)

    df_concat, formatted_data = compute_merge_results_to_dataframe(
        challenge_data=formatted_data,
        full_matches=full_matches,
        match_database=match_database,
        prefiltering_l2_threshold=prefiltering_l2_threshold)

    formatted_data.to_pickle(Paths.get_path_for_data_with_ids_pkl(language_embedding_model.get_name(),
                                                                  str(prefiltering_l2_threshold), "monkey"))

    df_concat.to_parquet(os.path.join("data","intermediate_data","reassemble_dataframe_output.parquet"))


