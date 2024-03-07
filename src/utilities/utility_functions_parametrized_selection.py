"""
This file contains the parametrized selection function which is used in the expert rules.
"""

# import packages/libraries:
import pandas as pd
from tqdm import tqdm

# get hardcoded potential partial duplicates tags of non overlapping words:

# pool of additional education information in the non overlapping words:
tag_education = ["degre", "scienc", "bachelor",
                 "master", "msc", "bsc", "diploma",
                 "school", "graduat", "phd", "physic",
                 "prerequisit", "secondari", "biolog", "math",
                 "pharmaci", "equiv", "scientif", "subject",
                 "engin", "informat", "comput", "electron",
                 "disciplin", "gymnast"]

# pool of additional contract information in the non overlapping words:
tag_contract = ["perman", "basi", "contract", "month",
                "fix", "fixedterm", "rang", "paid",
                "holiday", "pay", "option", "guarante",
                "temporari", "per", "hour", "durat",
                "eurmonth", "benefit", "annum", "salari", "net",
                "bonus", "maximum", "basic", "wage", "gross",
                "durat", "rate", "overtim", "czkmonth"]

# pool of additional schedule information in the non overlapping words:
tag_schedule = ["night", "friday", "hour", "long", "thursday", "noon",
                "monday", "schedul", "fulltim", "noon", "shift",
                "weekend", "weekday", "flexibl", "temporari", "sunday",
                "week", "day", "june", "juli", "schedul", "daytim", "august",
                "prorata", "time", "rata", "late", "daytim", "workload",
                "earli", "resourc", "human", "part", "parttim"]

# pool of additional technical skills information in the non overlapping words:
tag_tech_skills = ["methodolog", "web", "sql", "oracl", "javascript", "css",
                   "html", "scrum", "java", "microsoft", "aws", "azure",
                   "microsoft", "client", "salesforc", "python", "pyspark", "spark",
                   "oracle", "previous", "adob", "financ", "code", "quant"]

# pool of additional qualifications information in the non overlapping words:
tag_qualifications = ["drive", "licenc", "guild", "qualif", "essenti", "desir",
                      "option", "permiss", "profess",
                      "team", "precast", "basic", "architectur",
                      "leader", "research", "group", "fee", "proof",
                      "higher", "experi", "experienc", "group", "teach",
                      "level", "career", "experi", "least", "advanc",
                      "expert", "ideal", "profession", "sell",
                      "junior", "profici", "refurbish",
                      "prefer", "telephon", "call", "rotat",
                      "natur", "electt", "math", "level", "skill",
                      "intermedi", "organiz", "analyz",
                      "develop", "senior",
                      "certifi", "audit", "level",
                      "housekeep", "junior", "placement",
                      "nonbank", "plus", "solid",
                      "smoke", "pet", "techniqu", "medic",
                      "minimum", "colleagu"]

# pool of additional experience information in the non overlapping words:
tag_experience = ["year", "experi", "least", "alreadi", "hour",
                  "month", "saturdaynight", "sundaybank"]


def find_tags(text_input: str,
              tags: list) -> list:
    """
    Detects the non overlapping words in a text.

    :param str text_input: text to be checked
    :param list tags: tags to be used for the search
    :return list tags: list of non overlapping words in the text
    """

    # get a list of all words:
    words = [word for word in text_input.split()]

    # get the overlap:
    overlap = list(set(words) & set(tags))

    # return the overlap:
    return overlap


def parametrized_selection(
        df: pd.DataFrame,
        education_primary: int = 2, education_secondary: int = 1,
        contract_primary: int = 2, contract_secondary: int = 1,
        schedule_primary: int = 2, schedule_secondary: int = 1,
        tech_skills_primary: int = 2, tech_skills_secondary: int = 1,
        qualifications_primary: int = 2, qualifications_secondary: int = 1,
        experience_primary: int = 2, experience_secondary: int = 1) -> pd.DataFrame:
    """
    Performs the parametrized search in the non overlapping words and selects the ones, based on the given parameters.

    :param pd.DataFrame df:
    :param int education_primary: primary threshold for the education tags
    :param int education_secondary: secondary threshold for the education tags
    :param int contract_primary: primary threshold for the contract tags
    :param int contract_secondary: secondary threshold for the contract tags
    :param int schedule_primary: primary threshold for the schedule tags
    :param int schedule_secondary: secondary threshold for the schedule tags
    :param int tech_skills_primary: primary threshold for the technical skills tags
    :param int tech_skills_secondary: secondary threshold for the technical skills tags
    :param int qualifications_primary: primary threshold for the qualifications tags
    :param int qualifications_secondary: secondary threshold for the qualifications tags
    :param int experience_primary: primary threshold for the experience tags
    :param int experience_secondary: secondary threshold for the experience tags
    :return pd.DataFrame: dataframe with the output
    """

    tqdm.pandas(desc=f"Selecting based on tags")

    df["tag_education_overlap_1"] = df["string_non_overlap_1"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_education))
    df["tag_education_overlap_2"] = df["string_non_overlap_2"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_education))

    df["tag_contract_overlap_1"] = df["string_non_overlap_1"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_contract))
    df["tag_contract_overlap_2"] = df["string_non_overlap_2"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_contract))

    df["tag_schedule_overlap_1"] = df["string_non_overlap_1"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_schedule))
    df["tag_schedule_overlap_2"] = df["string_non_overlap_2"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_schedule))

    df["tag_tech_skills_overlap_1"] = df["string_non_overlap_1"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_tech_skills))
    df["tag_tech_skills_overlap_2"] = df["string_non_overlap_2"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_tech_skills))

    df["tag_qualifications_overlap_1"] = df["string_non_overlap_1"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_qualifications))
    df["tag_qualifications_overlap_2"] = df["string_non_overlap_2"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_qualifications))

    df["tag_experience_overlap_1"] = df["string_non_overlap_1"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_experience))
    df["tag_experience_overlap_2"] = df["string_non_overlap_2"].progress_apply(
        lambda x: find_tags(text_input=x, tags=tag_experience))

    df_tag_education = df[((df["tag_education_overlap_1"].apply(len) > education_primary) & (
            df["tag_education_overlap_2"].apply(len) <= education_secondary))
                          |
                          ((df["tag_education_overlap_2"].apply(len) > education_primary) & (
                                  df["tag_education_overlap_1"].apply(len) <= education_secondary))]

    df_tag_contract = df[((df["tag_contract_overlap_1"].apply(len) > contract_primary) & (
            df["tag_contract_overlap_2"].apply(len) <= contract_secondary))
                         |
                         ((df["tag_contract_overlap_2"].apply(len) > contract_primary) & (
                                 df["tag_contract_overlap_1"].apply(len) <= contract_secondary))]

    df_tag_schedule = df[((df["tag_schedule_overlap_1"].apply(len) > schedule_primary) & (
            df["tag_schedule_overlap_2"].apply(len) <= schedule_secondary))
                         |
                         ((df["tag_schedule_overlap_2"].apply(len) > schedule_primary) & (
                                 df["tag_schedule_overlap_1"].apply(len) <= schedule_secondary))]

    df_tag_tech_skills = df[((df["tag_tech_skills_overlap_1"].apply(len) > tech_skills_primary) & (
            df["tag_tech_skills_overlap_2"].apply(len) <= tech_skills_secondary))
                            |
                            ((df["tag_tech_skills_overlap_2"].apply(len) > tech_skills_primary) & (
                                    df["tag_tech_skills_overlap_1"].apply(len) <= tech_skills_secondary))]

    df_tag_qualifications = df[((df["tag_qualifications_overlap_1"].apply(len) > qualifications_primary) & (
            df["tag_qualifications_overlap_2"].apply(len) <= qualifications_secondary))
                               |
                               ((df["tag_qualifications_overlap_2"].apply(len) > qualifications_primary) & (
                                       df["tag_qualifications_overlap_1"].apply(len) <= qualifications_secondary))]

    df_tag_experience = df[((df["tag_experience_overlap_1"].apply(len) > experience_primary) & (
            df["tag_experience_overlap_2"].apply(len) <= experience_secondary))
                           |
                           ((df["tag_experience_overlap_2"].apply(len) > experience_primary) & (
                                   df["tag_experience_overlap_1"].apply(len) <= experience_secondary))]

    # combine all together:
    df_combined = pd.concat([df_tag_qualifications,
                             df_tag_education,
                             df_tag_schedule,
                             df_tag_tech_skills,
                             df_tag_contract,
                             df_tag_experience])

    # drop duplicates that might appear:
    df_combined = df_combined.drop_duplicates(subset=['id1', 'id2'])

    return df_combined
