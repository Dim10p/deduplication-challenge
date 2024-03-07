"""
This file contains the cleaning function which is applied on the raw data.
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
from src.utilities.utility_functions_cleaning import clean_strings, convert_str_date_to_int


def clean_input_dataset(input_path: str = None) -> None:
    """
    Cleans the raw data and saves the result as parquet.

    :param str input_path: the input path (defaults to the standard raw data input path)
    """

    # define the input path:
    input_data_path = Paths.RAW_DATA_PATH if not input_path else input_path
    print(f"Input data path: {input_data_path}")

    # read in the input data:
    print("Reading in the input data...")

    df = pd.read_csv(input_data_path,
                     index_col=None,
                     header=0,
                     engine='python',
                     encoding='utf-8')

    print(f"The file was read successfully!")

    print(f"The raw data contain {df.shape[0]} job advertisements in total.")

    # convert dtypes:
    df = df.convert_dtypes()

    # check dtypes:
    print("After importing the raw data, the following columns were discovered:")
    print(df.dtypes)
    # expected dtypes: string for all variables

    print("\nPerforming operations...")

    # detect, print, and drop rows with sketchy IDs:
    df[DataDimensions.ID] = pd.to_numeric(df[DataDimensions.ID], errors='coerce')
    print(f"\n{df[DataDimensions.ID].isna().sum()} sketchy IDs were found.")
    print(f"Dropping completely the rows with sketchy IDs (if any)...")
    df = df[df[DataDimensions.ID].notna()]
    df[DataDimensions.ID] = df[DataDimensions.ID].astype(int)
    print(f"The data now contain {df.shape[0]} job advertisements in total.")

    # detect and print the number of missing values:
    print(f"{df[DataDimensions.TITLE].isna().sum()} missing values were found in the titles.")
    print(f"{df[DataDimensions.COMPANY_NAME].isna().sum()} missing values were found in the company names.")
    print(f"{df[DataDimensions.DESCRIPTION].isna().sum()} missing values were found in the descriptions.")
    print(f"{df[DataDimensions.RETRIEVAL_DATE].isna().sum()} missing values were found in the retrieval dates.")
    print(f"{df[DataDimensions.LOCATION].isna().sum()} missing values were found in the locations.")
    print(f"{df[DataDimensions.COUNTRY_ID].isna().sum()} missing values were found in the country ids.")

    print("\nTurning the retrieval dates to integers...")
    # convert the retrieval date to integer:
    df[DataDimensions.RETRIEVAL_DATE] = df[DataDimensions.RETRIEVAL_DATE].apply(convert_str_date_to_int)

    # replace all missing values with "":
    print(f"\nReplacing missing values (if any) with ''...")
    df = df.fillna("")

    # clean company names, titles, and descriptions:
    print(f"\nStarting cleaning for company names, titles, and descriptions...\n")

    # create a new column with the clean company name:
    tqdm.pandas(desc="Cleaning company names")
    df[DataDimensions.COMPANY_NAME_CLEAN] = df[DataDimensions.COMPANY_NAME].progress_apply(
        lambda x: clean_strings(x,
                                deep=False,
                                lowercase=False))

    # create a new column with the clean title:
    tqdm.pandas(desc="Cleaning titles")
    df[DataDimensions.TITLE_CLEAN] = df[DataDimensions.TITLE].progress_apply(
        lambda x: clean_strings(x,
                                deep=False,
                                lowercase=False))

    # create a new column with the clean description:
    tqdm.pandas(desc="Cleaning descriptions")
    df[DataDimensions.DESCRIPTION_CLEAN] = df[DataDimensions.DESCRIPTION].progress_apply(
        lambda x: clean_strings(x,
                                deep=True,
                                lowercase=False))

    # print the number of the unique companies, before and after the cleaning:
    print(f"\nThere are {len(list(df[DataDimensions.COMPANY_NAME].unique()))} "
          f"unique company names.")
    # 15899
    print(
        f"There are {len(list(df[DataDimensions.COMPANY_NAME_CLEAN].unique()))} "
        f"unique company names after cleaning them.")
    # 15883

    # print the number of the unique titles, before and after the cleaning:
    print(f"There are {len(list(df[DataDimensions.TITLE].unique()))} "
          f"unique titles.")
    # 53192
    print(
        f"There are {len(list(df[DataDimensions.TITLE_CLEAN].unique()))} "
        f"unique titles after cleaning them.")
    # 52995

    # print the number of the unique descriptions, before and after the cleaning:
    print(f"There are {len(list(df[DataDimensions.DESCRIPTION].unique()))} "
          f"unique descriptions.")
    # 59022
    print(
        f"There are {len(list(df[DataDimensions.DESCRIPTION_CLEAN].unique()))} "
        f"unique descriptions after cleaning them.")
    # 58968

    # create additional helper columns:
    print(f"Creating additional helper columns...")

    # get a unique hash per unique clean description:
    df[DataDimensions.CLEAN_DESCRIPTION_HASH] = df.groupby(DataDimensions.DESCRIPTION_CLEAN)[
        DataDimensions.ID].transform(
        'ngroup')

    # get a unique hash per unique clean title:
    df[DataDimensions.CLEAN_TITLE_HASH] = df.groupby(DataDimensions.TITLE_CLEAN)[
        DataDimensions.ID].transform(
        'ngroup')

    # convert dtypes again:
    df = df.convert_dtypes()
    print("\nBefore saving intermediate result the following columns and data types are available:")
    # print dtypes:
    print(df.dtypes)

    # define the output path:
    output_path = Paths.FORMATTED_DATA_PATH

    # export the output as parquet:
    df.to_parquet(output_path)

    print(f"\nSaved to {output_path}.")
