"""
This file contains the embedding models creation function.
"""

# import packages/libraries:
import os
import os.path
from os import path

# import SentenceTransformer object:
from sentence_transformers import SentenceTransformer

# import constants:
from src.utilities.constants import Paths


class EmbeddingModel:

    # define the models original full names:
    models = {'distiluse': 'distiluse-base-multilingual-cased',
              'minilm': 'paraphrase-multilingual-MiniLM-L12-v2',
              'xlm': 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
              'mpnet': 'all-mpnet-base-v2'}

    def __init__(self, model: str, max_seq_length: int):
        """
        :param str model: select one between 'distiluse', 'minilm', 'xlm' or 'mpnet'
        """
        self._max_seq_length = max_seq_length
        self._name = self._validate_model_choice(model)
        self._model_path = Paths.get_language_model_path(self._name)
        self._model = self._create_model(self._name)

    def encode(self, *args, **kwargs):
        return self._model.encode(*args, **kwargs)

    def get_name(self):
        return self._name

    def get_model_path(self):
        return self._model_path

    def model(self):
        return self._model

    def _create_model(self, name: str) -> SentenceTransformer:

        # define the path where the models will be downloaded:
        # TODO: replace os.getcwd() with a generic root path to '*\\DeduplicationChallenge directory'
        model_folder = os.path.join(os.getcwd(), self.get_model_path())
        if path.exists(model_folder):
            folder = os.listdir(model_folder)[0]
            model = SentenceTransformer(os.path.join(model_folder, folder))
        else:
            print("Model is not downloaded yet. Downloading...")
            model = SentenceTransformer(self.models.get(name), cache_folder=self.get_model_path())

        # set model max_seq_length equal to 512 to work with larger strings:
        model.max_seq_length = self._max_seq_length
        return model

    def _validate_model_choice(self, model_name: str) -> str:
        if model_name not in self.models:
            raise ValueError(
                f"{model_name} is not available. Please supply one of one of: {str(list(self.models.keys()))}")
        else:
            return model_name
