import pandas as pd
import pytest
from src.embedding_models import EmbeddingModel
from src.embedding_process.index_search import perform_index_search
from src.utilities.constants import DataDimensions

distiluse = EmbeddingModel('distiluse', max_seq_length=100)
minilm = EmbeddingModel('minilm', 100)
xlm = EmbeddingModel('xlm', 100)
mpnet = EmbeddingModel('mpnet', 100)

EMBEDDING_MODELS = [mpnet, distiluse, minilm, xlm]


def compute_output(input_data, language_model):
    # create a corpus of all descriptions (list):
    description_corpus = list(input_data[DataDimensions.TITLE_AND_DESCRIPTION].unique())  # unique not needed

    # start embedding (locally, this takes 2-3 hours on the whole dataset):
    corpus_embedding = (language_model.encode(
        description_corpus, convert_to_tensor=True, show_progress_bar=True)
    )

    results = perform_index_search(corpus_embedding, input_data, language_model.get_name(), n_matches=2)
    return results


def get_input_data(n_title: int,
                   n_text: int,
                   text_1: str = "the quick brown fox jumps over the lazy dog. ",
                   text_2: str = "it was the best of times, it was the worst of time. ",
                   title_1: str = "system administrator. ",
                   title_2: str = "system administrator. ",
                   print_length: bool = False,
                   ) -> pd.DataFrame:
    """
    Prepares input data frame containing two rows of title/text multiples

    :param print_length: whether to print the length
    :param title_2: title of second row
    :param title_1: title of first row
    :param text_2: text of second row
    :param text_1: text of first row
    :param n_title: number of times the title is to be repeated
    :param n_text: number of times the text to be repeated
    :return:
    """
    descriptions = [
        item1 := title_1 * n_title + text_1 * n_text,
        item2 := title_2 * n_title + text_2 * n_text
    ]
    if print_length:
        print(f"Longest input text is {max(len(item1), len(item2))} characters long")

    hashes = list(range(0, len(descriptions)))

    input_data = pd.DataFrame(data={DataDimensions.TITLE_AND_DESCRIPTION_HASH: hashes,
                                    DataDimensions.TITLE_AND_DESCRIPTION: descriptions})

    return input_data


@pytest.mark.slow
@pytest.mark.parametrize("language_model", EMBEDDING_MODELS)
def test_whether_different_input_length_of_equivalent_information_gives_same_results(language_model: EmbeddingModel):
    input_data_short = get_input_data(n_title=1, n_text=1)
    results_short = compute_output(input_data_short, language_model)

    input_data_long = get_input_data(n_title=150, n_text=150, print_length=True)
    results_long = compute_output(input_data_long, language_model)

    assert results_short == results_long, "identical information is not treated idential if text is too long"


@pytest.mark.slow
@pytest.mark.parametrize("language_model", EMBEDDING_MODELS)
def test_whether_different_order_in_sentence_gets_same_result(language_model: EmbeddingModel):
    input_data_short = get_input_data(
        n_title=1,
        n_text=1,
        text_1="the quick brown fox jumps over the lazy dog it was the best of times, it was the worst of time.",
        text_2="it was the best of times, it was the worst of time the quick brown fox jumps over the lazy dog.")
    results_short = compute_output(input_data_short, language_model)

    assert round(results_short[0][1], 10) == round(results_short[1][0], 10), "sentence order changed results"
