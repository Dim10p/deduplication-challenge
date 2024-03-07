"""
This file contains all the utility/cleaning functions that are used in this project.

References:
- https://docs.python.org/3/library/unittest.html
"""

# import necessary libraries/modules:
import unittest

# import all utility functions that will be tested:
from src.utilities.utility_functions_cleaning import (remove_html_tags,
                                                      convert_html_to_ascii,
                                                      convert_to_lowercase,
                                                      keep_only_ascii_and_punc,
                                                      remove_whitespaces,
                                                      split_uppercase_after_lowercase,
                                                      remove_repeated_punctuations,
                                                      clean_strings)


# define all unit tests:

class TestRemoveHtmlTags(unittest.TestCase):
    """
    Test class for the remove_html_tags function.
    """

    def test_empty_string(self):
        input_data = ""
        expected_outcome = input_data
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_no_html_string(self):
        input_data = "Cases without !html. tags"
        expected_outcome = input_data
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_balanced_html_string(self):
        input_data = "<br>This is a test</br>"
        expected_outcome = "This is a test"
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_sequential_html_string(self):
        input_data = "<test>hello</test><br>This is a test</br>"
        expected_outcome = "hello This is a test"
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_empty_brackets_string(self):
        input_data = "hola<>hello"
        expected_outcome = input_data
        actual_outcome = remove_html_tags(input_data)  # the leftover punctuations will go away later
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_html_mix_case_1(self):
        input_data = "<test>hello<test"
        expected_outcome = "hello<test"  # the leftover punctuations will go away later
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_html_mix_case_2(self):
        input_data = "<<test>>hello"
        expected_outcome = "< >hello"  # the leftover punctuations will go away later
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_html_mix_case_3(self):
        input_data = "<<br>just testing here >hello"
        expected_outcome = "< just testing here >hello"  # the leftover punctuations will go away later
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_html_mix_case_4(self):
        input_data = "<test<br>hello"
        expected_outcome = "<test hello"  # the leftover punctuations will go away later
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_html_mix_case_5(self):
        input_data = """<a title=">">"""
        expected_outcome = """">"""
        actual_outcome = remove_html_tags(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here


class TestConvertHtmlToAscii(unittest.TestCase):
    """
    Test class for the convert_html_to_ascii convert_html_to_ascii function.
    """

    def test_greater_than(self):
        input_data = "&gt;"
        expected_outcome = ">"
        actual_outcome = convert_html_to_ascii(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_greater_than_with_adjacent_text(self):
        input_data = "30 &gt; 10"
        expected_outcome = "30 > 10"
        actual_outcome = convert_html_to_ascii(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_greater_than_with_adjacent_text_without_white_spaces(self):
        input_data = "30&gt;10"
        expected_outcome = "30>10"
        actual_outcome = convert_html_to_ascii(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_greater_than_with_additional_and_symbol(self):
        input_data = "30&&gt;10"
        expected_outcome = "30&>10"
        actual_outcome = convert_html_to_ascii(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here


class TestConvertToLowercase(unittest.TestCase):
    """
    Test class for the convert_html_to_ascii convert_to_lowercase function.
    """

    def test_simple_conversion(self):
        input_data = "This is jUst a SIMPLE test To Check if this function is !< working properly.10/01/T"
        expected_outcome = "this is just a simple test to check if this function is !< working properly.10/01/t"
        actual_outcome = convert_to_lowercase(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here


class TestKeepOnlyAsciiAndPunc(unittest.TestCase):
    """Test class for the convert_html_to_ascii keep_only_ascii_and_punc function.
    ***More characters can be found here: https://terpconnect.umd.edu/~zben/Web/CharSet/htmlchars.html
    """

    def test_no_text_to_replace(self):
        input_data = "hello this is text"
        expected_outcome = input_data
        actual_outcome = keep_only_ascii_and_punc(input_data, keep="some_punc")
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_inverted_exclamation_mark(self):
        input_data = "hello this is text¡"
        expected_outcome = "hello this is text"
        actual_outcome = keep_only_ascii_and_punc(input_data, keep="some_punc")
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_latin_small_letter_u_with_grave(self):
        input_data = "hello this is textù"
        expected_outcome = "hello this is textù"
        actual_outcome = keep_only_ascii_and_punc(input_data, keep="some_punc")
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_latin_small_letter_a_with_diaeresis(self):
        input_data = "hello this is textä"
        expected_outcome = "hello this is textä"
        actual_outcome = keep_only_ascii_and_punc(input_data, keep="some_punc")
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_random_non_ascii_character(self):
        input_data = "hello this is text×"
        expected_outcome = "hello this is text"
        actual_outcome = keep_only_ascii_and_punc(input_data, keep="some_punc")
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_non_accepted_punctuations(self):
        input_data = """!"#$%&'()*This is a random text+,-./\:;<=>?@[]^`{|}~123 Hello testers."""
        expected_outcome = "! % This is a random text , ./\\: 123 Hello testers."
        actual_outcome = keep_only_ascii_and_punc(input_data, keep="some_punc")
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here


class TestRemoveWhitespaces(unittest.TestCase):
    """
    Test class for the convert_html_to_ascii remove_whitespaces function.
    """

    def test_leading_and_trailing_whitespaces(self):
        input_data = " hello this is text "
        expected_outcome = "hello this is text"
        actual_outcome = remove_whitespaces(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_multiple_leading_and_trailing_whitespaces(self):
        input_data = "            hello this is text          "
        expected_outcome = "hello this is text"
        actual_outcome = remove_whitespaces(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_multiple_leading_and_trailing_and_extra_whitespaces(self):
        input_data = "            hello    this     is text          "
        expected_outcome = "hello this is text"
        actual_outcome = remove_whitespaces(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_multiple_tabs_and_newlines(self):
        input_data = "            \t    hello   \n \n    this   \t      is text    \n      "
        expected_outcome = "hello this is text"
        actual_outcome = remove_whitespaces(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here


class TestSplitUppercaseAfterLowercase(unittest.TestCase):
    """
    Test class for the convert_html_to_ascii remove_whitespaces function.
    """

    def test_case_1(self):
        input_data = "Hello thisIs a text"
        expected_outcome = "Hello this Is a text"
        actual_outcome = split_uppercase_after_lowercase(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_case_2(self):
        input_data = "Hello thisIsNot a Text"
        expected_outcome = "Hello this Is Not a Text"
        actual_outcome = split_uppercase_after_lowercase(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here


class TestRemoveRepeatedPunctuations(unittest.TestCase):
    """
    Test class for the remove_repeated_punctuations convert_to_lowercase function.
    """

    def test_remove_repeated_dots(self):
        input_data = "Hello.. this is.... A test"
        expected_outcome = "Hello. this is. A test"
        actual_outcome = remove_repeated_punctuations(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_remove_repeated_commas(self):
        input_data = "Hello, this is,,,,,,,, A test"
        expected_outcome = "Hello, this is, A test"
        actual_outcome = remove_repeated_punctuations(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_remove_repeated_exclamation_marks(self):
        input_data = "Hello!!!!!! this is! A test!!!!!"
        expected_outcome = "Hello! this is! A test!"
        actual_outcome = remove_repeated_punctuations(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

    def test_remove_repeated_percentage_symbols(self):
        input_data = "%%Hello%%% this is A test"
        expected_outcome = "%Hello% this is A test"
        actual_outcome = remove_repeated_punctuations(input_data)
        self.assertEqual(expected_outcome, actual_outcome)  # add assertion here

