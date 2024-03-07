"""
This file contains all the translation functions that are used in this project.
"""

# import packages/libraries:
from googletrans import Translator

# initialize a Translator object once:
translator = Translator()

# set maximum number of attempts in case of API fail:
MAX_ATTEMPTS = 5


def translate(text: str) -> str:
    """
    Translates a string using Google API translator.

    :param str text: the input string
    :return str modified_text: the translated string
    """

    for attempt in range(MAX_ATTEMPTS):
        try:
            return translator.translate(text, dest='en').text  # always translate to english
        except Exception:
            pass
    return "ERROR WHILE TRANSLATING"
