"""
This file contains the perfect deduplication function which is applied on the raw data.
"""

# import packages/libraries:
import itertools
import warnings

# add the simplefilter before importing pandas in order to work:
warnings.simplefilter(action='ignore',
                      category=FutureWarning)
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
from tqdm import tqdm
import pandas as pd

# activate tqdm pandas:
tqdm.pandas()

# import constants and functions:
from src.utilities.constants import Paths, DataDimensions, OutputDataDims, OutputDataFields
from src.utilities.utility_functions_cleaning import convert_str_date_to_int


def find_perfect_duplicates(input_path: str = None) -> None:
    """
    Finds the perfect matches in the raw data and saves the intermediate results in csv.
    Generates all FULL duplicates and part of the (full) TEMPORAL duplicates.

    :param str input_path: path of the raw data (defaults to the standard raw data input path)
    """

    # define the input path:
    input_path = Paths.RAW_DATA_PATH if not input_path else input_path
    print(f"Input data path: {input_path}")

    # read in the input data:
    print("Reading in the input data...")

    df = pd.read_csv(input_path,
                     index_col=None,
                     header=0,
                     engine='python',
                     encoding='utf-8')

    print(f"The file was read successfully!")

    print(f"The raw data contain {df.shape[0]} job advertisements in total.")

    print("\nPerforming operations...")

    # convert dtypes:
    df = df.convert_dtypes()

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
    df[DataDimensions.RETRIEVAL_DATE] = df[DataDimensions.RETRIEVAL_DATE].apply(convert_str_date_to_int)

    # replace all missing values with "":
    print(f"\nReplacing missing values (if any) with ''...")
    df = df.fillna("")

    print(f"\nDetecting FULL and (full) TEMPORAL duplicates in terms of retrieval date, title, and description...")

    # create , title and description combined column:
    df[DataDimensions.TITLE_AND_DESCRIPTION] = (
            df[DataDimensions.TITLE] + "<>" +
            df[DataDimensions.DESCRIPTION]
    )

    # map id to dates:
    id_to_date = df[[DataDimensions.ID, DataDimensions.RETRIEVAL_DATE]]

    # number each group from 0 to the total number of groups (cumcount):
    df[DataDimensions.FULL_MATCH_ID] = df.groupby(DataDimensions.TITLE_AND_DESCRIPTION).ngroup()
    df[DataDimensions.FULL_MATCH_ID] = pd.to_numeric(df[DataDimensions.FULL_MATCH_ID], errors='coerce')

    # create a column with a list of all full duplicates ids:
    df = (df.filter([DataDimensions.ID, DataDimensions.FULL_MATCH_ID])
          .groupby(DataDimensions.FULL_MATCH_ID)[DataDimensions.ID]
          .apply(list)
          .reset_index(name='full_duplicates_ids'))

    # create a column with the length of the lists:
    df['total_number'] = df['full_duplicates_ids'].apply(lambda x: len(x))

    # keep only rows with at least one duplicate:
    df = df[df["total_number"] > 1]

    # create tuples of all unique double combinations inside the list:
    df["double_combinations"] = df["full_duplicates_ids"].apply(
        lambda x: list(itertools.combinations_with_replacement(x, 2)))

    # explode all combinations to new rows:
    df = df.explode("double_combinations")

    # sort each combination in ascending order:
    df["double_combinations"] = df["double_combinations"].apply(lambda x: list(sorted(x, key=int)))

    # expand combinations to new columns:
    df[[OutputDataDims.ID1, OutputDataDims.ID2]] = pd.DataFrame(df.double_combinations.tolist(), index=df.index)

    # keep only rows where ID1 < ID2:
    df = df[df[OutputDataDims.ID1] < df[OutputDataDims.ID2]]

    # merge and separate between FULL and TEMPORAL duplicates combinations:
    df = (df.merge(id_to_date.rename(columns={DataDimensions.ID: DataDimensions.ID + "1",
                                              DataDimensions.RETRIEVAL_DATE: DataDimensions.RETRIEVAL_DATE + "1"}),
                   on=OutputDataDims.ID1)
          .merge(id_to_date.rename(columns={DataDimensions.ID: DataDimensions.ID + "2",
                                            DataDimensions.RETRIEVAL_DATE: DataDimensions.RETRIEVAL_DATE + "2"}),
                 on=OutputDataDims.ID2)
          )

    df_full_dups = pd.DataFrame(df, copy=True)
    df_temp_dups = pd.DataFrame(df, copy=True)
    df_full_dups = df_full_dups[
        df_full_dups[DataDimensions.RETRIEVAL_DATE + "2"] == df_full_dups[DataDimensions.RETRIEVAL_DATE + "1"]]
    df_temp_dups = df_temp_dups[
        df_temp_dups[DataDimensions.RETRIEVAL_DATE + "2"] != df_temp_dups[DataDimensions.RETRIEVAL_DATE + "1"]]

    # create new column equal to FULL everywhere:
    df_full_dups[OutputDataDims.TYPE] = OutputDataFields.Type.FULL

    # create new column equal to TEMPORAL everywhere:
    df_temp_dups[OutputDataDims.TYPE] = OutputDataFields.Type.TEMPORAL

    # combine both:
    df = pd.concat([df_temp_dups, df_full_dups])

    # keep only necessary columns:
    df = df.filter([OutputDataDims.ID1, OutputDataDims.ID2, OutputDataDims.TYPE])

    # force to int type conversion:
    df[OutputDataDims.ID1] = df[OutputDataDims.ID1].astype(int)
    df[OutputDataDims.ID2] = df[OutputDataDims.ID2].astype(int)

    # print number of final output:
    print(f"""The following number of duplicate combinations were found:
     {df.groupby(OutputDataDims.TYPE)[OutputDataDims.ID1].count()}""")

    # define the export path:
    output_path = Paths.FULL_DUPLICATES_PATH

    # export as temporary parquet with the first full and temporal duplicates:
    df.to_parquet(output_path,
                  index=False)

    print(f"\nSaved at {output_path}.")
