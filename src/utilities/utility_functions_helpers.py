"""
This file contains all the helper functions that are used in this project.
"""

# import necessary libraries/modules:
import os

# import constants:
from src.utilities.constants import Directories, Colours

# allow colours to be printed in command prompt and powershell natively:
os.system("")


def create_required_folders() -> None:
    """
    Creates the necessary directories for the project.
    """

    # create all the necessary directories locally if they do not exist already:
    for name, path in Directories.DataDirectories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(
                Colours.GREEN +
                f"The directory for the {name} has been created successfully at {path}."
                + Colours.RESET)
        else:
            print(
                Colours.RED +
                f"The directory for the {name} already exists at {path}." +
                Colours.RESET)


