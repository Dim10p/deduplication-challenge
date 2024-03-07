"""
This file contains the data processing functions which are used in this project.
"""

# import packages/libraries:
import copy
from typing import Union, List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

# import constants:
from src.utilities.constants import DataDimensions, OutputDataFields, OutputDataDims


def explode_into_final_format(df: pd.DataFrame, match_type: str) -> pd.DataFrame:
    map_of_column_names = {
        DataDimensions.SEMANTIC_MATCHES: OutputDataFields.Type.SEMANTIC,
        DataDimensions.TEMPORAL_MATCHES: OutputDataFields.Type.TEMPORAL,
    }

    df = (df.set_index(DataDimensions.ID)[match_type]
          .explode().reset_index()
          .rename(columns={DataDimensions.ID: OutputDataDims.ID1,
                           match_type: OutputDataDims.ID2}))
    df[OutputDataDims.TYPE] = map_of_column_names[match_type]
    df = df[~df[OutputDataDims.ID2].isna()]

    return df


# TODO(jannic): consider refactoring this to avoid the flag and do the sum of the two lists in the semantics case
# in the dataframe rather than here
def assign_semantic_or_partial_matches_to_id_via_description_hashes(
        identical_cleaned_descriptions_ids: Union[List[int], None],
        description_hash: int,
        description_hash_to_ids: Dict[int, List[int]],
        semantic_or_partial_matches: Dict[int, List[int]],
        partial_match: bool = False) -> List[int]:
    """
    :param identical_cleaned_descriptions_ids: the list of IDs that anyway have the same cleaned description
    :param int description_hash:  the hash of that description
    :param dict description_hash_to_ids:  the dictionary that maps the hashes to the list of their ids
    :param dict semantic_or_partial_matches: the match database (either semantic or partial)
    :param bool partial_match: boolean flag whether semantic or partial
    :return list: output list
    """

    # handle nans (i.e. cases where there no perfect duplicates on the cleaned description):
    if not isinstance(identical_cleaned_descriptions_ids, list):
        identical_cleaned_descriptions_ids = []

    # get the description hashes:
    cleaned_description_hash_semantic_matches = list(semantic_or_partial_matches.get(description_hash).keys())

    respective_ids_in_dataframe = []

    # for each of those, find the original IDs that they belong to. note that this can be multiple ones!
    for hash_id in cleaned_description_hash_semantic_matches:
        respective_ids_in_dataframe = respective_ids_in_dataframe + description_hash_to_ids.get(int(hash_id), [])

    if partial_match:
        return list(set(respective_ids_in_dataframe))
    else:
        return list(set(identical_cleaned_descriptions_ids + respective_ids_in_dataframe))


def create_dict_of_semantic_and_temporal_vs_partial_matches(
        match_database: Dict[int, Dict[int, float]],
        lower_threshold: float = 0.3
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Based on the matches for the hashed cleaned descriptions we compute:
    - semantic matches
    - temporal matches

    :param dict match_database:
    :param float lower_threshold:
    :return tuple: tuple output
    """

    semantic_and_temporal_matches_copy = copy.deepcopy(match_database)

    semantic_and_temporal_matches = dict()

    # create a dict with only semantic matches (i.e. dropping matches with too large l2 distances)
    pbar = tqdm(semantic_and_temporal_matches_copy.items())
    pbar.set_description("Find semantic and temporal matches")
    for hash_id, matches in pbar:
        semantic_matches_for_this_hash_id = dict()
        for matched_hash_id, l2_of_matched_hash in matches.items():
            if l2_of_matched_hash < lower_threshold:
                semantic_matches_for_this_hash_id.update({matched_hash_id: l2_of_matched_hash})

        semantic_and_temporal_matches.update({int(hash_id): semantic_matches_for_this_hash_id})
        del hash_id, semantic_matches_for_this_hash_id, matches

    return semantic_and_temporal_matches


def add_semantic_and_temporal_matches_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function splits the list of semantic_and_temporal_matches columns into two separate columns taking
    into account the retrieval date

    :param pd.DataFrame df: dataframe to be processed
    :return pd.DataFrame: output dataframe
    """

    # temporal duplicates are semantic duplicates with varying advertisement retrieval dates!

    id_to_retrieval_date = (
        df.filter([DataDimensions.ID, DataDimensions.RETRIEVAL_DATE])
        .set_index(DataDimensions.ID)
        .to_dict()
        .get(DataDimensions.RETRIEVAL_DATE))

    id_to_semantic_and_temporal_matches = (
        df.filter([DataDimensions.ID, DataDimensions.SEMANTIC_AND_TEMPORAL_MATCHES])
        .set_index(DataDimensions.ID)
        .to_dict()
        .get(DataDimensions.SEMANTIC_AND_TEMPORAL_MATCHES))

    # get the data structures to store the results:
    semantic_matches, temporal_matches = dict(), dict()

    pbar = tqdm(id_to_semantic_and_temporal_matches.items())
    pbar.set_description("Defining semantic and temporal matches column")
    for job_id, semantic_and_temporal_matches_job_ids in pbar:
        # for job_id x this is the retrieval data:
        retrieval_date_of_job_id = id_to_retrieval_date[job_id]

        # if there are matches, they are stored here:
        semantic_matches_for_this_job_id = []
        temporal_matches_for_this_job_id = []
        # for each match job id, compare the retrieval date...
        for semantic_or_temporal_matches_job_id in semantic_and_temporal_matches_job_ids:
            # ...if they match, this is a semantic_match.
            retrieval_date_of_this_one_match = id_to_retrieval_date[semantic_or_temporal_matches_job_id]

            if retrieval_date_of_this_one_match == retrieval_date_of_job_id:

                semantic_matches_for_this_job_id.append(semantic_or_temporal_matches_job_id)

            else:

                temporal_matches_for_this_job_id.append(semantic_or_temporal_matches_job_id)

            # ...if not, this is a temporal match.

        # no point in being matched to itself:
        if job_id in semantic_matches_for_this_job_id:
            semantic_matches_for_this_job_id.remove(job_id)

        semantic_matches.update({job_id: semantic_matches_for_this_job_id})
        temporal_matches.update({job_id: temporal_matches_for_this_job_id})

        del semantic_matches_for_this_job_id, temporal_matches_for_this_job_id

    del id_to_retrieval_date, id_to_semantic_and_temporal_matches

    def set_to_column(job_id: int,
                      mapping_of_job_id_to_matches: Dict[int, List[int]]):
        return mapping_of_job_id_to_matches[job_id]

    tqdm.pandas(desc="Assign semantic matches to columns")
    df[DataDimensions.SEMANTIC_MATCHES] = df.progress_apply(
        lambda row: set_to_column(job_id=row[DataDimensions.ID],
                                  mapping_of_job_id_to_matches=semantic_matches),
        axis=1)

    tqdm.pandas(desc="Assign temporal matches to columns")
    df[DataDimensions.TEMPORAL_MATCHES] = df.progress_apply(
        lambda row: set_to_column(job_id=row[DataDimensions.ID],
                                  mapping_of_job_id_to_matches=temporal_matches),
        axis=1)

    return df


def keep_best_match_per_id1_id2_combination(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each groupby_cols combinations, keep the row where value_column is the lowest.

    :param pd.DataFrame df: input dataframe
    :return pd.DataFrame: output dataframe
    """

    if len(df) == 0:
        return df

    use_most_granular = {
        OutputDataFields.Type.FULL: 1,
        OutputDataFields.Type.SEMANTIC: 2,
        OutputDataFields.Type.TEMPORAL: 3,
        OutputDataFields.Type.PARTIAL: 4
    }

    df["keep_smallest_of_those"] = df[OutputDataDims.TYPE].map(use_most_granular)
    result = df.sort_values("keep_smallest_of_those").groupby([OutputDataDims.ID1, OutputDataDims.ID2],
                                                              as_index=False).first()
    del result["keep_smallest_of_those"]  # no longer needed

    return result
