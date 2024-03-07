"""
This file contains the preparation of input data for the embedding function which is applied on the cleaned data.
It also translates the data, or not, based on the parameter.
"""

# import necessary libraries/modules:
import warnings

# add before import pandas as pd in order to work:
warnings.simplefilter(action='ignore',
                      category=FutureWarning)
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
import pandas as pd
from tqdm import tqdm

# activate tqdm pandas:
tqdm.pandas()

# import constants and functions:
from src.utilities.constants import Paths, DataDimensions
from src.utilities.utility_functions_translations import translate
from src.utilities.utility_functions_cleaning import merge_title_and_description_with_weights


def prepare_input_data_for_embedding(
        input_path: str = None,
        title_weight: float = 0.3,
        translation: bool = True) -> None:
    """
    Translates the cleaned data if translation parameter is set to True else continues with the clean data,
    and merges title and description with the desired weight on the title.

    :param str input_path: the input path (defaults to the standard raw data input path)
    :param float title_weight: the weight of the title in the column the embedding model is to be run on
    :param bool translation: select to translate the data or not (defaults to True)
    """

    # define the input path:
    input_data_path = Paths.FORMATTED_DATA_PATH if not input_path else input_path
    print(f"Formatted data path: {input_data_path}")

    # read in the input data:
    print("Reading in the formatted data...")

    df = pd.read_parquet(input_data_path)

    print(f"The file was read successfully!")

    # if translation is set to True, translate titles and descriptions:
    if translation:
        print(f"\nStarting translations...\n")

        # get all unique descriptions with their hash:
        df_desc = df.drop_duplicates(DataDimensions.CLEAN_DESCRIPTION_HASH, keep='first')

        # get all unique titles with their hash:
        df_title = df.drop_duplicates(DataDimensions.CLEAN_TITLE_HASH, keep='first')

        # translate all unique descriptions:
        tqdm.pandas(desc="Translating descriptions")
        df_desc[DataDimensions.DESCRIPTION_TRANSLATED] = df_desc[DataDimensions.DESCRIPTION_CLEAN].progress_apply(
            lambda x: translate(x))

        # translate all unique titles:
        tqdm.pandas(desc="Translating titles")
        df_title[DataDimensions.TITLE_TRANSLATED] = df_title[DataDimensions.TITLE_CLEAN].progress_apply(
            lambda x: translate(x))

        # store the translations as parquet:
        df_desc.to_parquet(Paths.TRANSLATED_DESCRIPTION_PATH)
        df_title.to_parquet(Paths.TRANSLATED_TITLE_PATH)

        # merge the translated description back to the dataframe:
        df = df.merge(df_desc[[DataDimensions.CLEAN_DESCRIPTION_HASH, DataDimensions.DESCRIPTION_TRANSLATED]],
                      on=DataDimensions.CLEAN_DESCRIPTION_HASH,
                      how='left')

        # merge the translated title back to the dataframe:
        df = df.merge(df_title[[DataDimensions.CLEAN_TITLE_HASH, DataDimensions.TITLE_TRANSLATED]],
                      on=DataDimensions.CLEAN_TITLE_HASH,
                      how='left')

    else:
        df[DataDimensions.TITLE_TRANSLATED] = df[DataDimensions.TITLE_CLEAN]
        df[DataDimensions.DESCRIPTION_TRANSLATED] = df[DataDimensions.DESCRIPTION_CLEAN]

    # create a new column merging the title with weight and the description:
    tqdm.pandas(desc="Creating new merged column (weighted title & description)")
    df[DataDimensions.TITLE_AND_DESCRIPTION] = df.progress_apply(
        lambda row: merge_title_and_description_with_weights(
            row[DataDimensions.TITLE_TRANSLATED if translation else DataDimensions.TITLE_CLEAN],
            row[DataDimensions.DESCRIPTION_TRANSLATED if translation else DataDimensions.DESCRIPTION_CLEAN],
            desired_weight_on_title=title_weight),
        axis=1)

    # get a unique hash per unique combined title and description:
    df[DataDimensions.TITLE_AND_DESCRIPTION_HASH] = df.groupby(DataDimensions.TITLE_AND_DESCRIPTION)[
        DataDimensions.ID].transform(
        'ngroup')

    # convert dtypes:
    df = df.convert_dtypes()
    print("\nBefore saving translated result the following columns and data types are available:")
    # print dtypes:
    print(df.dtypes)

    # define the output path:
    output_path = Paths.INTERMEDIATE_DATA_PATH

    # export output as parquet:
    df.to_parquet(output_path)

    print(f"\nSaved to {output_path}.")
