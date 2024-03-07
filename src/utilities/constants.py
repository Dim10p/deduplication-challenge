"""
This file contains all the constants that are used in this project.
"""

# import necessary libraries/modules:
import os


class Directories(object):

    #  dictionary with all data directories that need to be created:
    DataDirectories: dict = {
        "input data": os.path.join("data", "input_data"),
        "intermediate data": os.path.join("data", "intermediate_data"),
        "output data": os.path.join("data", "output_data")
    }


class Paths(object):

    # raw/input data path:
    RAW_DATA_PATH: str = os.path.join("data", "input_data", "wi_dataset.csv")

    # intermediate data paths:
    FORMATTED_DATA_PATH: str = os.path.join("data", "intermediate_data", "formatted_data.parquet")
    TRANSLATED_DESCRIPTION_PATH: str = os.path.join("data", "intermediate_data", "translated_descriptions.parquet")
    TRANSLATED_TITLE_PATH: str = os.path.join("data", "intermediate_data", "translated_titles.parquet")
    INTERMEDIATE_DATA_PATH: str = os.path.join("data", "intermediate_data", "intermediate_data.parquet")

    # embedded data paths:
    EMBEDDING_PATH: str = os.path.join("data", "intermediate_data", "embedding.pt")
    EMBEDDING_PATH_SAMPLE: str = os.path.join("data", "intermediate_data", "embedding_sample.pt")

    # pre-filtered data path:
    FINAL_PRE_FILTERING_PATH: str = os.path.join("data", "intermediate_data", "final_pre_filtering.parquet")

    @classmethod
    def get_embedding_path(cls, model_name: str) -> str:
        return os.path.join("data", "intermediate_data", f"embedding_{model_name}.pt")

    @classmethod
    def get_embedding_path_by_variable(cls, model_name: str, variable_name: str) -> str:
        return os.path.join("data", "intermediate_data", f"embedding_{model_name}_{variable_name}.pt")

    @classmethod
    def get_hash2variable_path(cls, variable_name: str) -> str:
        return os.path.join("data", "intermediate_data", f"map_{variable_name}_to_hash.parquet")

    @classmethod
    def get_language_model_path(cls, model_name: str) -> str:
        return os.path.join("data", "models", f"embedding_{model_name}")

    # mapping unique descriptions with hash paths:
    MAP_DESCRIPTION_TO_HASH: str = os.path.join(
        "data", "intermediate_data", "map_description_to_hash.parquet")
    MAP_DESCRIPTION_TO_HASH_WITH_MATCHES: str = os.path.join(
        "data", "intermediate_data", "map_description_to_hash_with_matches.parquet")

    @classmethod
    def get_path_for_map_description_to_hash_with_matches(cls, model_name: str) -> str:
        return os.path.join("data", "intermediate_data", f"map_description_to_hash_with_matches_{model_name}.pickle")

    # data with ids path:
    DATA_WITH_IDS_PKL: str = os.path.join("data", "intermediate_data", "data_with_ids.pickle")

    @classmethod
    def get_path_for_data_with_ids_pkl(cls,
                                       model_name: str,
                                       lower_threshold_semantic_matches: str,
                                       lower_threshold_partial_match: str) -> str:
        path: str = os.path.join(
            "data",
            "intermediate_data",
            f"data_with_ids_{model_name}_{lower_threshold_semantic_matches}_{lower_threshold_partial_match}.pickle"
        )
        return path

    # fully duplicated data path:
    FULL_DUPLICATES_PATH: str = os.path.join("data", "intermediate_data", "full_duplicates.parquet")

    # final/output data:
    @classmethod
    def get_path_for_final_data(cls,
                                model_name: str,
                                lower_threshold_semantic_matches: str,
                                lower_threshold_partial_match: str) -> str:
        path: str = os.path.join(
            "data",
            "intermediate_data",
            f"duplicates_{model_name}_{lower_threshold_semantic_matches}_{lower_threshold_partial_match}.parquet"
        )
        return path

    # output/final data path:
    FINAL_DATA: str = os.path.join("data", "output_data", "duplicates.csv")


class OutputDataDims(object):
    # data dimensions for the output/final data:
    ID1 = "id1"
    ID2 = "id2"
    TYPE = "type"
    ID1_ID2 = [ID1, ID2]


class OutputDataFields(object):
    class Type(object):
        # types of duplicates to be defined in the output/final data:
        FULL = "FULL"
        SEMANTIC = "SEMANTIC"
        TEMPORAL = "TEMPORAL"
        PARTIAL = "PARTIAL"


class DataDimensions(object):
    """
    Collects all string constants for the dataframe columns.
    """

    # original columns names:
    ID: str = 'id'
    TITLE: str = "title"
    DESCRIPTION: str = "description"
    LOCATION: str = "location"
    COUNTRY_ID: str = "country_id"
    COMPANY_NAME: str = "company_name"
    RETRIEVAL_DATE: str = "retrieval_date"

    # new columns names str type:
    TITLE_AND_DESCRIPTION: str = "title_and_description"
    DESCRIPTION_CLEAN: str = DESCRIPTION + "_clean"
    DESCRIPTION_CLEAN_LENGTH: str = DESCRIPTION_CLEAN + "_length"
    DESCRIPTION_CLEAN_HASH: str = DESCRIPTION + "_clean_hash"
    DESCRIPTION_CLEAN_WITHOUT_TITLE: str = DESCRIPTION + "_clean_without_title"
    COMPANY_NAME_CLEAN: str = COMPANY_NAME + "_clean"
    COMPANY_NAME_CLEAN_HASH: str = COMPANY_NAME + "_clean_hash"
    TITLE_CLEAN: str = TITLE + "_clean"
    TITLE_CLEAN_HASH = TITLE_CLEAN + "_hash"
    DESCRIPTION_TRANSLATED: str = DESCRIPTION + "_translated"
    TITLE_TRANSLATED: str = TITLE + "_translated"
    DESCRIPTION_CLEAN_LANGUAGES: str = DESCRIPTION_CLEAN + "_languages"
    IDENTICAL_CLEANED_DESCRIPTIONS_IDS: str = "identical_cleaned_descriptions"
    SEMANTIC_AND_TEMPORAL_MATCHES: str = 'semantic_and_temporal_matches'
    SEMANTIC_MATCHES: str = 'semantic_matches'
    TEMPORAL_MATCHES: str = 'temporal_matches'
    PARTIAL_MATCHES: str = 'partial_matches'

    # new columns number type:
    FULL_MATCH_ID: str = 'full_match_id'
    DESCRIPTION_LENGTH: str = DESCRIPTION + "_length"
    CLEAN_DESCRIPTION_HASH: str = DESCRIPTION + "_hash"
    CLEAN_TITLE_HASH: str = TITLE + "_hash"
    TITLE_AND_DESCRIPTION_HASH: str = TITLE_AND_DESCRIPTION + "_hash"
    COMPANY_ID: str = "company_id"
    TITLE_ID: str = "title_id"
    NUMBER_OF_DESCRIPTIONS_PER_COMPANY: str = "number_of_descriptions_per_company"
    NUMBER_OF_POSTINGS_PER_TITLE: str = "number_of_postings_per_title"

    # hash matches:
    @classmethod
    def hash_match_number(cls, number: int):
        return f"hash_match_{number}"

    @classmethod
    def hash_match_number_l2(cls, number: int):
        return f"hash_match_{number}_l2"

    @classmethod
    def get_hash_name_of_variable(cls, variable_name: str) -> str:
        return f"{variable_name}_hash"


class Colours(object):
    # colours codes:
    BLACK: str = '\033[30m'
    RED: str = '\033[31m'
    GREEN: str = '\033[32m'
    YELLOW: str = '\033[33m'
    BLUE: str = '\033[34m'
    MAGENTA: str = '\033[35m'
    CYAN: str = '\033[36m'
    WHITE: str = '\033[37m'
    UNDERLINE: str = '\033[4m'
    RESET: str = '\033[0m'
