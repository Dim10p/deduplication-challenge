import gc
import itertools
import os
import pickle
from functools import lru_cache

from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utilities.utility_functions_parametrized_selection import parametrized_selection
from src.utilities.utility_functions_data_processing import keep_best_match_per_id1_id2_combination

tqdm.pandas()
from src.utilities.constants import DataDimensions
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

from src.utilities.constants import OutputDataDims, OutputDataFields, Paths

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from functools import lru_cache

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

@lru_cache(maxsize=None)
def fuzzy_ratio(text_1: str, text_2: str) -> float:
    """computes the fuzzy ratio of two texts"""
    ratio = fuzz.token_set_ratio(text_1, text_2)
    return ratio / 100

@lru_cache(maxsize=None)
def not_matching_string(text_a="", text_b=""):
    text_a = text_a.translate(str.maketrans("", "", string.punctuation + string.digits))
    text_b = text_b.translate(str.maketrans("", "", string.punctuation + string.digits))
    stemmed_words = [stemmer.stem(word) for word in text_a.split()]
    # Remove stop words
    stemmed_words = [word for word in stemmed_words if word not in stop_words]
    # Join the remaining words into a string
    stemmed_sentence_a = " ".join(stemmed_words)

    stemmed_words = [stemmer.stem(word) for word in text_b.split()]
    # Remove stop words
    stemmed_words = [word for word in stemmed_words if word not in stop_words]
    # Join the remaining words into a string
    stemmed_sentence_b = " ".join(stemmed_words)
    words_a = set(stemmed_sentence_a.lower().split())
    words_b = set(stemmed_sentence_b.lower().split())
    # Find words unique to each sentence
    unique_words_a = words_a.difference(words_b)
    unique_words_b = words_b.difference(words_a)
    isEmpty_a = (unique_words_a == set())
    if isEmpty_a:
        unique_words_string_a = ""
    else:
        unique_words_string_a = " ".join(unique_words_a)

    isEmpty_b = (unique_words_b == set())
    if isEmpty_b:
        unique_words_string_b = ""
    else:
        unique_words_string_b = " ".join(unique_words_b)

    return unique_words_string_a, unique_words_string_b


def number_words(text: str = "") -> int:
    """Returns the number of words"""
    text_list = set(text.lower().split())
    length = len(text_list)
    return length


def find_l2_between_two_hash_ids(hash_id1: int, hash_id_2: int, match_database: dict) -> float:
    """helper function to find the l2 based on hashes"""
    l2 = match_database.get(hash_id1).get(hash_id_2)
    return l2


def find_similarity_score_of_two_columns(
        df: pd.DataFrame,
        column_to_find_pairs_on: str,
        embedding_model: SentenceTransformer
) -> pd.DataFrame:
    columns = [column_to_find_pairs_on + "_1", column_to_find_pairs_on + "_2"]
    # create unique text vectors for embedding
    unique_texts = list(set(list(itertools.chain.from_iterable([df[column].to_list() for column in columns]))))
    unique_texts_hash_map = pd.DataFrame()
    unique_texts_hash_map['text'] = unique_texts
    unique_texts_hash_map = unique_texts_hash_map.reset_index(names='unique_text_hash_id')

    for column in columns:
        df = df.merge(unique_texts_hash_map.rename(
            columns={'unique_text_hash_id': f"{column}_hashid"}),
            left_on=column,
            right_on='text')
        del df['text']

    embedding = embedding_model.encode(unique_texts_hash_map['text'].to_list(),
                                       convert_to_tensor=True,
                                       show_progress_bar=True)

    # run embedding on unique text vectors
    @lru_cache(maxsize=embedding.shape[0])
    def find_similarity_for_one_pair(hash_1: int, hash_2: int, embedding) -> float:
        cosine_scores = util.cos_sim(embedding[hash_1], embedding[hash_2])
        cosine_scores = cosine_scores[0][0]
        cosine_scores = float(cosine_scores.cpu().numpy())
        return cosine_scores

    tqdm.pandas(desc=f"Compute cosine for {columns}")
    df[f"mpnet_{column_to_find_pairs_on}_similarity_score"] = df.filter(
        [f"{column_to_find_pairs_on}_1_hashid", f"{column_to_find_pairs_on}_2_hashid"]).progress_apply(
        lambda row: find_similarity_for_one_pair(
            hash_1=row[f"{column_to_find_pairs_on}_1_hashid"],
            hash_2=row[f"{column_to_find_pairs_on}_2_hashid"],
            embedding=embedding),
        axis=1)

    del df[f"{column_to_find_pairs_on}_1_hashid"]
    del df[f"{column_to_find_pairs_on}_2_hashid"]

    return df


def prepare_input_for_expert_rules(language_embedding_model: SentenceTransformer,
                                  description_similarities: bool):
    """prepare input dataset for expert rules"""

    # read in data:
    df = pd.read_parquet(os.path.join("data", "intermediate_data", "reassemble_dataframe_output.parquet"))

    # attach all information, excluding L2:
    data = pd.read_parquet(Paths.INTERMEDIATE_DATA_PATH)
    df1 = data.copy()
    df2 = data.copy()
    df1.columns = [col + "_1" for col in list(df1.columns)]
    df2.columns = [col + "_2" for col in list(df2.columns)]
    df = df.rename(columns={'id1': 'id_1', 'id2': 'id_2'})
    df = df.merge(df1, on='id_1', how='inner')
    df = df.merge(df2, on='id_2', how='inner')
    del data, df1, df2
    gc.collect()

    # filter:
    df = df.filter(
        ['id_1', 'id_2', 'type', 'retrieval_date_1', 'retrieval_date_2', 'location_1', 'location_2', 'country_id_1',
         'country_id_2', 'company_name_1', 'company_name_2', 'title_1', 'title_2', 'description_1', 'description_2',
         'title_clean_1', 'title_clean_2', 'description_clean_1', 'description_clean_2', 'title_translated_1',
         'title_translated_2', 'description_translated_1', 'description_translated_2'])

    # match back L2:
    formatted_data = pd.read_parquet(Paths.INTERMEDIATE_DATA_PATH)

    # get the results
    with open(Paths.get_path_for_map_description_to_hash_with_matches(language_embedding_model.get_name()), 'rb') as f:
        match_database = pickle.load(f)

    df = df.rename(columns={'id_1': 'id1', 'id_2': 'id2'})

    id_2_hash = formatted_data[[DataDimensions.ID, DataDimensions.TITLE_AND_DESCRIPTION_HASH]]
    id_2_hash[DataDimensions.ID] = pd.to_numeric(id_2_hash[DataDimensions.ID], errors='coerce')
    df = (
        df.merge(id_2_hash.rename(
            columns={DataDimensions.ID: OutputDataDims.ID1,
                     DataDimensions.TITLE_AND_DESCRIPTION_HASH: DataDimensions.TITLE_AND_DESCRIPTION_HASH + "1"}),
            on=OutputDataDims.ID1)
            .merge(id_2_hash.rename(
            columns={DataDimensions.ID: OutputDataDims.ID2,
                     DataDimensions.TITLE_AND_DESCRIPTION_HASH: DataDimensions.TITLE_AND_DESCRIPTION_HASH + "2"}),
            on=OutputDataDims.ID2)
    )

    df['l2'] = df.progress_apply(
        lambda row: find_l2_between_two_hash_ids(row[DataDimensions.TITLE_AND_DESCRIPTION_HASH + "1"],
                                                 row[DataDimensions.TITLE_AND_DESCRIPTION_HASH + "2"],
                                                 match_database),
        axis=1
    )

    # create additional columns:
    df['same_title'] = 0
    df.loc[(df.title_1 == df.title_2), 'same_title'] = 1

    # get string lengths:
    df['description_str_len_1'] = df['description_1'].str.len()
    df['description_str_len_2'] = df['description_2'].str.len()
    df['description_clean_str_len_1'] = df['description_clean_1'].str.len()
    df['description_clean_str_len_2'] = df['description_clean_2'].str.len()
    df['description_translated_str_len_1'] = df['description_translated_1'].str.len()
    df['description_translated_str_len_2'] = df['description_translated_2'].str.len()

    # get string diffs:
    df['description_str_diff'] = abs(df['description_str_len_1'] - df['description_str_len_2'])
    df['description_clean_str_diff'] = abs(df['description_clean_str_len_1'] - df['description_clean_str_len_2'])
    df['description_translated_str_diff'] = abs(
        df['description_translated_str_len_1'] - df['description_translated_str_len_2'])

    # get similarities for titles:
    df = find_similarity_score_of_two_columns(df, "title_translated", language_embedding_model)

    # get similarities for descriptions if True:
    if description_similarities:
        df = find_similarity_score_of_two_columns(df, "description_translated", language_embedding_model)

    df = df.convert_dtypes()

    # final rules (based on submission_6.py)
    df = df.drop(labels=['description_1', 'description_2', 'description_clean_1', 'description_clean_2'], axis=1)

    df.to_parquet(Paths.FINAL_PRE_FILTERING_PATH)


def apply_expert_rules_on_data(
        partial_rule: str = 'tag',
        temporal_title_similarity_unrestricted: float = 0.8,
        temporal_title_similarity_restricted: float = 0.92,
        temporal_l2_threshold: float = 0.1):
    """Apply expert rules"""

    df = pd.read_parquet(Paths.FINAL_PRE_FILTERING_PATH)

    full_matches = pd.read_parquet(Paths.FULL_DUPLICATES_PATH)

    df_temporals = df[df[OutputDataDims.TYPE] == OutputDataFields.Type.TEMPORAL]
    df_temporals = df_temporals.merge(
        full_matches[[OutputDataDims.ID1, OutputDataDims.ID2]],
        on=[OutputDataDims.ID1, OutputDataDims.ID2],
        how='outer',
        indicator=True)

    print("Constructing temporal duplicates based on expert rules")
    df_temporals = df_temporals[df_temporals['_merge'] == "left_only"]
    del df_temporals["_merge"]

    df_temporals[
        'description_translated_2_contains_visidarbi'] = df_temporals.description_translated_2.str.lower().str.contains(
        r'visidarbi.lv')
    df_temporals[
        'description_translated_1_contains_visidarbi'] = df_temporals.description_translated_1.str.lower().str.contains(
        r'visidarbi.lv')

    temporals_group_4a = df_temporals[
        ((df_temporals['description_translated_2_contains_visidarbi'])
         &
         (df_temporals['description_translated_1_contains_visidarbi'])
         &
         (df_temporals.mpnet_title_translated_similarity_score < 0.85))]

    temporals_group_4b = df_temporals[
        (((df_temporals['description_translated_2_contains_visidarbi'])
          &
          (~df_temporals['description_translated_1_contains_visidarbi']))
         |
         ((df_temporals['description_translated_1_contains_visidarbi'])
          &
          (~df_temporals['description_translated_2_contains_visidarbi'])))]

    temporals_group_4c = df_temporals[((df_temporals.country_id_1 != df_temporals.country_id_2) &
                                       ((df_temporals.country_id_1 != '') & (df_temporals.country_id_2 != '')))]

    temporals_group_4 = pd.concat([temporals_group_4a, temporals_group_4b, temporals_group_4c])
    del temporals_group_4a, temporals_group_4b, temporals_group_4c
    temporals_group_4 = temporals_group_4.drop_duplicates(subset=OutputDataDims.ID1_ID2, keep='first')

    # Exclude temporals_group_4 from df_temporals
    df_temporals = df_temporals.merge(temporals_group_4[OutputDataDims.ID1_ID2], on=OutputDataDims.ID1_ID2, how='left',
                                      indicator=True)
    df_temporals._merge.value_counts()
    df_temporals = df_temporals[df_temporals['_merge'] == 'left_only']
    df_temporals = df_temporals.drop('_merge', axis=1)

    # Exclude another "bad" group - "Group 3"
    temporals_group_3 = df_temporals[(df_temporals.mpnet_title_translated_similarity_score < 0.41) |
                                     (df_temporals.l2 > 0.21)]

    df_temporals = df_temporals.merge(
        temporals_group_3[OutputDataDims.ID1_ID2],
        on=OutputDataDims.ID1_ID2, how='left',
        indicator=True)

    df_temporals = df_temporals[df_temporals['_merge'] == 'left_only']
    df_temporals = df_temporals.drop('_merge', axis=1)
    del temporals_group_3

    # temporals group 1: pick the best ones
    temporals_group_1 = df_temporals[
        ((df_temporals.company_name_1 == df_temporals.company_name_2)
         &
         ((df_temporals.company_name_1 != '') & (df_temporals.company_name_2 != '')))
        & (df_temporals.mpnet_title_translated_similarity_score > temporal_title_similarity_unrestricted)]

    # exclude group_1
    df_temporals = df_temporals.merge(temporals_group_1[OutputDataDims.ID1_ID2], on=OutputDataDims.ID1_ID2,
                                      how='outer', indicator=True)
    df_temporals._merge.value_counts()
    df_temporals = df_temporals[df_temporals['_merge'] == 'left_only']
    df_temporals = df_temporals.drop('_merge', axis=1)

    # Alternatively, df_temporals.mpnet_description_translated_similarity_score > 0.92 as 2nd condition
    temporals_group_2 = df_temporals[
        (df_temporals.mpnet_title_translated_similarity_score > temporal_title_similarity_restricted)
        & (df_temporals.l2 < temporal_l2_threshold)]

    final_temporals = pd.concat([temporals_group_1, temporals_group_2])
    del temporals_group_1, temporals_group_2, temporals_group_4

    print("Constructing semantic duplicates based on expert rules")
    df_semantics = df[df[OutputDataDims.TYPE] == OutputDataFields.Type.SEMANTIC]

    df_semantics[
        'description_translated_2_contains_visidarbi'] = df_semantics[
        'description_translated_2'].str.lower().str.contains(
        r'visidarbi.lv')
    df_semantics[
        'description_translated_1_contains_visidarbi'] = df_semantics[
        'description_translated_1'].str.lower().str.contains(
        r'visidarbi.lv')

    semantics_group_4a = df_semantics[
        ((df_semantics['description_translated_2_contains_visidarbi'])
         &
         (df_semantics['description_translated_1_contains_visidarbi'])
         &
         (df_semantics.mpnet_title_translated_similarity_score < 0.85))
    ]

    semantics_group_4b = df_semantics[
        (df_semantics['description_translated_2_contains_visidarbi']
         &
         (~df_semantics['description_translated_1_contains_visidarbi']))
        |
        (df_semantics['description_translated_1_contains_visidarbi']
         &
         (~df_semantics['description_translated_2_contains_visidarbi']))
        ]

    semantics_group_4c = df_semantics[((df_semantics.country_id_1 != df_semantics.country_id_2) &
                                       ((df_semantics.country_id_1 != '') & (df_semantics.country_id_2 != '')))]

    semantics_group_4 = pd.concat([semantics_group_4a, semantics_group_4b, semantics_group_4c])
    del semantics_group_4a, semantics_group_4b, semantics_group_4c
    semantics_group_4 = semantics_group_4.drop_duplicates(subset=OutputDataDims.ID1_ID2, keep='first')

    # kick out Semantics group4
    df_semantics = df_semantics.merge(semantics_group_4[OutputDataDims.ID1_ID2],
                                      on=OutputDataDims.ID1_ID2,
                                      how='outer',
                                      indicator=True)
    df_semantics = df_semantics[df_semantics['_merge'] == 'left_only']
    df_semantics = df_semantics.drop('_merge', axis=1)

    df_empty_location = df_semantics[((df_semantics.location_1 == "") & (df_semantics.location_2 != ""))
                                     |
                                     ((df_semantics.location_2 == "") & (df_semantics.location_1 != ""))]

    df_empty_company = df_semantics[((df_semantics.company_name_1 == "") & (df_semantics.company_name_2 != ""))
                                    |
                                    ((df_semantics.company_name_2 == "") & (df_semantics.company_name_1 != ""))]

    df_empty_info = pd.concat([df_empty_location, df_empty_company])
    df_empty_info = df_empty_info.drop_duplicates(subset=OutputDataDims.ID1_ID2, keep='first')

    df_semantics = df_semantics.merge(df_empty_info[OutputDataDims.ID1_ID2], on=OutputDataDims.ID1_ID2,
                                      how='outer', indicator=True)
    df_semantics = df_semantics[df_semantics['_merge'] == 'left_only']
    df_semantics = df_semantics.drop('_merge', axis=1)

    # Partial duplicates

    # generate string ratio:
    df_semantics["string_ratio"] = 2 * df_semantics["description_translated_str_diff"] / (
        df_semantics["description_translated_str_len_1"] + df_semantics["description_translated_str_len_2"])

    if partial_rule == 'fuzzy':

        tqdm.pandas(desc="fuzzywuzzy overlap")
        df_semantics["fuzzy_ratio"] = df_semantics.progress_apply(
            lambda row: fuzzy_ratio(text_1=row["description_translated_1"],
                                    text_2=row["description_translated_2"]),
            axis=1)

        partials_fuzzy_string = df_semantics[(df_semantics["fuzzy_ratio"] > 0.90) &
                                             (df_semantics['string_ratio'] > 0.04) &
                                             (df_semantics['description_translated_str_diff'] > 40)]

    elif partial_rule == 'tag':

        # search for non-overlap in texts
        tqdm.pandas(desc="detect non overlapping descriptions")
        df_semantics[["string_non_overlap_1", "string_non_overlap_2"]] = df_semantics.progress_apply(
            lambda row: not_matching_string(text_a=row["description_translated_1"],
                                            text_b=row["description_translated_2"]),
            axis=1, result_type="expand")

        tqdm.pandas(desc="counting non overlaps")
        df_semantics["words_non_overlap_1"] = df_semantics.progress_apply(
            lambda row: number_words(text=row["string_non_overlap_1"]), axis=1)
        df_semantics["words_non_overlap_2"] = df_semantics.progress_apply(
            lambda row: number_words(text=row["string_non_overlap_2"]), axis=1)

        partials_overlap = df_semantics[
            ((df_semantics.string_non_overlap_1 == "") &
             (df_semantics.string_non_overlap_2 != "") &
             (df_semantics.words_non_overlap_2 > 1))
            |
            ((df_semantics.string_non_overlap_2 == "") & (
                    df_semantics.string_non_overlap_1 != "") & (
                     df_semantics.words_non_overlap_1 > 1))
            ]

        partials_tags = parametrized_selection(
            df_semantics,
            education_primary=2, education_secondary=1,
            contract_primary=2, contract_secondary=1,
            schedule_primary=2, schedule_secondary=1,
            tech_skills_primary=2, tech_skills_secondary=1,
            qualifications_primary=2, qualifications_secondary=1,
            experience_primary=2, experience_secondary=1)

    elif partial_rule == 'string_ratio':

        partials_string_ratio = df_semantics[(df_semantics['string_ratio'] > 0.05) &
                                             (df_semantics['description_translated_str_diff'] > 50)]

    else:
        raise NotImplementedError("partial_rule must be set to 'tag', 'fuzzy' or 'string_ratio'")

    if partial_rule == 'fuzzy':
        final_partials = (partials_fuzzy_string
                          .drop_duplicates(subset=['id1', 'id2'],
                                           keep='first')
                          )

    elif partial_rule == 'tag':
        final_partials = (pd.concat([partials_tags, partials_overlap])
            .drop_duplicates(
            subset=['id1', 'id2'],
            keep='first')
        )

    elif partial_rule == 'string_ratio':
        final_partials = (partials_string_ratio
            .drop_duplicates(
            subset=['id1', 'id2'],
            keep='first')
        )

    else:
        raise NotImplementedError("partial_rule must be set to 'tag' or 'fuzzy' or 'string_ratio'")

    final_partials = final_partials.filter(['id1', 'id2'])

    final_partials = final_partials.drop_duplicates(subset=OutputDataDims.ID1_ID2, keep='first')
    final_partials[OutputDataDims.TYPE] = OutputDataFields.Type.PARTIAL

    # drop partials from semantics
    df_semantics = df_semantics.merge(

        final_partials[OutputDataDims.ID1_ID2],
        on=OutputDataDims.ID1_ID2,
        how='outer',
        indicator=True)

    df_semantics = df_semantics[df_semantics['_merge'] == 'left_only']
    del df_semantics['_merge']

    print("Putting it all together...")
    # concat everything
    df_concat = pd.DataFrame(pd.concat(
        [
            df_semantics.filter(OutputDataDims.ID1_ID2 + [OutputDataDims.TYPE]),
            final_partials.filter(OutputDataDims.ID1_ID2 + [OutputDataDims.TYPE]),
            final_temporals.filter(OutputDataDims.ID1_ID2 + [OutputDataDims.TYPE]),
            pd.read_parquet(Paths.FULL_DUPLICATES_PATH)
        ]
    ))

    df_concat[OutputDataDims.ID1] = df_concat[OutputDataDims.ID1].astype(int)
    df_concat[OutputDataDims.ID2] = df_concat[OutputDataDims.ID2].astype(int)

    # keep the most granular for each
    df_concat = keep_best_match_per_id1_id2_combination(df_concat)

    print("Saving output")
    # make sure to only export cases where ID1 < ID2:
    df_concat = df_concat[df_concat[OutputDataDims.ID1] < df_concat[OutputDataDims.ID2]]

    df_concat.to_parquet(os.path.join("data", "intermediate_data", "final_output_from_prod_version.parquet"))
    df_concat.to_csv(Paths.FINAL_DATA, index=False, header=False)


def apply_expert_rules(
        language_embedding_model: SentenceTransformer,
        partial_rule: str,
        temporal_title_similarity_unrestricted: float,
        temporal_title_similarity_restricted: float,
        temporal_l2_threshold: float,
        description_similarities: bool = False):
    """Wrapper to build input data for expert rules and apply expert rules"""

    prepare_input_for_expert_rules(language_embedding_model, description_similarities)

    apply_expert_rules_on_data(
        partial_rule=partial_rule,
        temporal_title_similarity_unrestricted=temporal_title_similarity_unrestricted,
        temporal_title_similarity_restricted=temporal_title_similarity_restricted,
        temporal_l2_threshold=temporal_l2_threshold
    )
