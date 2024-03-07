"""
This file contains all the cleaning functions that are used in this project.

References:
- https://swatimeena989.medium.com/beginners-guide-for-preprocessing-text-data-f3156bec85ca#2fb9
- https://www.quora.com/Can-word-embeddings-like-Glove-Fasttext-and-word2vec-consider-punctuation-and-capitalization
- https://www.enjoyalgorithms.com/blog/text-data-pre-processing-techniques-in-ml
- https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
- https://www.quora.com/How-does-word2vec-work-Can-someone-walk-through-a-specific-example/answer/Ajit-Rajasekharan
- https://www.enjoyalgorithms.com/blog/text-data-pre-processing-techniques-in-ml
- https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html
- https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/
- https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches/
- https://resources.experfy.com/ai-ml/text-preprocessing-for-nlp-and-machine-learning-tasks/
- https://towardsdatascience.com/regex-essential-for-nlp-ee0336ef988d
- https://studymachinelearning.com/text-preprocessing-removal-of-punctuations/
- https://datascience.stackexchange.com/questions/33381/would-keeping-all-punctuation-make-any-sense-in-word2vec
- https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/
- https://predictivehacks.com/tokenizer-for-nlp-tasks/
- https://www.sbert.net/examples/applications/parallel-sentence-mining/README.html
- https://ijece.iaescore.com/index.php/IJECE/article/view/25456/15384
"""

# import necessary libraries/modules:
import re
import html
from math import ceil
from random import randint


def merge_title_and_description_with_weights(title: str,
                                             description: str,
                                             desired_weight_on_title: float) -> str:
    """
    Merges the title and the description.
    However, while the titles are much shorter than the descriptions (on average), they are more important in a way.
    There is a two-step process.

    :param str title: the title string
    :param str description: the description string
    :param float desired_weight_on_title: the weight to be applied on the title
    :return str merged: the merged string of the weighted combination
    """

    # get title and description strings lengths:
    len_title = len(title)
    len_description = len(description)

    # get the corresponding share of the title relative to the description:
    share_title = len_title / (len_description + len_title)
    # round down to get a multiplier:
    inflate_factor = max(ceil(desired_weight_on_title / share_title), 1)

    # create the merged string:
    merged = inflate_factor * (title + " ") + description

    # returned the merged string:
    return merged


def convert_str_date_to_int(text: str) -> int:
    """
    Takes 'YYYY-MM-DD' date strings and returns YYYYMMDD as an integer.
    In case the values are missing values, returns a random RRRRRRRR integer.

    :param str text: a date in 'YYYY-MM-DD' format
    :return int result: the respective date integer
    """

    try:
        result = int(text.replace("-", ""))
    except:
        result = randint(10_000_000, 10_000_000 * 2)

    return result


def remove_html_tags(text) -> str:
    """
    Removes all the HTML tags from a given string (e.g, <br>, <strong>).

    References:
    - https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
    - https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string

    :param str text: the input string
    :return str modified_text: the clean string
    """

    # define pattern that removes html tags:
    html_tags_removal = re.compile('<[^><]+>')

    # remove all html tags and replace with whitespace:
    modified_text = re.sub(html_tags_removal, ' ', text)

    # remove unnecessary whitespaces:
    modified_text = ' '.join(modified_text.split())

    # return clean text:
    return modified_text


def convert_html_to_ascii(text: str) -> str:
    """
    Replaces all HTML character references with the equivalent ASCII characters.

    References:
    - https://www.educative.io/answers/what-is-htmlunescape-in-python

    :param str text: the input string
    :return str modified_text: the clean string
    """

    # convert all html character references to ascii:
    modified_text = html.unescape(text)

    # return clean text:
    return modified_text


def convert_to_lowercase(text: str) -> str:
    """
    Converts all characters to lowercase.

    :param str text: the input string
    :return str modified_text: the clean string
    """

    # convert all characters to lowercase:
    modified_text = text.lower()

    # return clean text:
    return modified_text


def keep_only_ascii_and_punc(text: str,
                             keep: str) -> str:
    """
    Performs two operations:
        1)Keeps only valid ASCII characters & a set of punctuations in the string.
        2)Replaces strings that have no ASCII characters at all with "".

    References:
    - https://www.educative.io/answers/remove-all-the-punctuation-marks-from-a-sentence-using-regex
    - https://www.quora.com/Why-should-punctuation-be-removed-in-Word2vec

    :param str text: the input string
    :param str keep: the type of pattern to use
    :return str modified_text: the clean string
    """

    # define a pattern that accepts all ascii and all punctuations:
    accept_ascii_and_all_punc = re.compile(r'[^ \!\"\#\$\%\&\\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\\\]\^\`\{\|\}\~\w]+')

    # define a pattern that accepts all ascii and specific punctuations:
    # allow for the following punctuations: ! % \ , . / :
    accept_ascii_and_specific_punc = re.compile(r'[^ \!\%\\\,\.\/\:\\\\\w]+')

    # define a pattern that accepts all ascii and no punctuations:
    accept_only_ascii = re.compile(r'[^ \w]+')

    # define a dictionary with all patterns:
    pat_dict = {"all_punc": accept_ascii_and_all_punc,
                "some_punc": accept_ascii_and_specific_punc,
                "no_punc": accept_only_ascii}

    # replace all non-accepted characters with a whitespace:
    modified_text = re.sub(pat_dict[keep], ' ', text)

    # remove extra/unnecessary whitespaces:
    modified_text = ' '.join(modified_text.split())

    # define a pattern with ascii characters only (no whitespace):
    ascii_chars_removal = re.compile(r'[^\w]+')

    # replace string with '' if the string has only punctuations or whitespaces:
    if re.sub(ascii_chars_removal, '', modified_text) == '':
        modified_text = ''

    # return clean text:
    return modified_text


def remove_whitespaces(text: str) -> str:
    """
    Performs two operations:
        1)Removes all leading and trailing whitespaces.
        2)Removes all extra/unnecessary whitespaces between characters.

    References:
    - https://stackoverflow.com/questions/1546226/is-there-a-simple-way-to-remove-multiple-spaces-in-a-string
    - https://stackoverflow.com/questions/10711116/strip-spaces-tabs-newlines-python

    :param str text: the input string
    :return str modified_text: the clean string
    """

    # remove all leading and trailing whitespaces, new lines and tabs:
    modified_text = text.strip()

    # define pattern that removes extra whitespaces, new lines and tabs between characters:
    # standard approach:
    # extra_whitespaces_removal = re.compile(' +')
    # adjusted approach:
    extra_whitespaces_removal = re.compile('\s+')

    # remove all extra whitespaces:
    modified_text = re.sub(extra_whitespaces_removal, ' ', modified_text)

    # return clean text:
    return modified_text


def split_uppercase_after_lowercase(text: str) -> str:
    """
    Adds a whitespace between lowercase characters that are followed directly by an uppercase character.
    *** however, this can be risky, in case of specific company names or acronyms e.g, StudiJob!

    :param str text: the input string
    :return str modified_text: the clean string
    """

    # if the text is equal to "":
    if text == "":
        # return the same text:
        return text
    else:
        # define an empty string:
        modified_text = ""

        # loop over all characters of the string:
        for i in range(len(text) - 1):
            # if a lowercase character is followed by an uppercase character:
            if text[i].islower() and text[i + 1].isupper():
                # add a whitespace after the first character and append:
                modified_text += text[i] + " "
            else:
                # do not perform operation and append text:
                modified_text += text[i]
        # add the last character of the string:
        modified_text += text[-1]

    # return clean text:
    return modified_text


def remove_repeated_punctuations(text: str) -> str:
    """
    Replaces multiple consecutive fullstops, commas and exclamation marks with one.

    :param str text: the input string
    :return str modified_text: the clean string
    """

    # replace multiple consecutive punctuations with one:
    modified_text = re.sub(r'([^\w\s])\1+', r'\1', text)

    # return clean text:
    return modified_text


def clean_strings(text: str,
                  deep: bool = True,
                  lowercase: bool = True) -> str:
    """
    Wrapper function:
    Cleans a string field using all the cleaning functions.
    It can be applied on descriptions, titles and company names differently.

    :param str text: the input string
    :param bool deep: choose to apply deep cleaning or not (defaults to True)
    :param bool lowercase: choose to lowercase in the end (defaults to True)
    :return str clean_text: the final clean string
    """

    # apply all cleaning functions sequentially, depending on the field that will be cleaned:
    clean_text = convert_html_to_ascii(text)  # apply everywhere
    clean_text = remove_html_tags(clean_text)  # apply everywhere
    if deep:
        clean_text = split_uppercase_after_lowercase(clean_text)  # apply to descriptions only
    clean_text = keep_only_ascii_and_punc(clean_text, keep="all_punc")  # apply everywhere
    clean_text = remove_repeated_punctuations(clean_text)  # apply everywhere
    clean_text = remove_whitespaces(clean_text)  # apply everywhere
    if deep:
        if lowercase:
            clean_text = convert_to_lowercase(clean_text)  # apply to descriptions only

    # return clean text:
    return clean_text


def contains_only_punctuations_regex(input_string: str) -> bool:
    """
    Returns True if the input string contains only punctuation characters, else False.

    :param str input_string: string to apply the functionality
    :return bool: True if the text contains only punctuations, else False
    """

    return bool(re.match(r'^[^\w]+$', input_string))
