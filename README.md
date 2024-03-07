[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
# Deduplication Challenge
This repository tracks the codes for the [data deduplication](https://statistics-awards.eu/competitions/4#learn_the_details) challenge hosted by the Jožef Stefan Institute for Eurostat.

The main file is a jupyter notebook that calls all other methods: [`main.ipynb`](main.ipynb).


The dataset made available by Eurostat includes the following fields:
 - **ID**: The unique OJA identifier. It is attributed by the Organisation committee.
 - **Job title. The title of the job, retrieved directly from the website.
 - **Job description.**: The description of the online job advertisement, retrieved directly from the website.
 - **Job location**: The place where the job is to be held, extracted automatically from the job description by the WIH.
 - **Country ID**: The country ID of the job position derived from the job location.
 - **Company name**: The name of the company that is hiring, extracted from the job description.
 - **Advertisement retrieval date**: The date (day, month, year) when the job advertisement was retrieved by the web bots of the WIH.

The dataset contains the following types of duplicates:

 - **Full duplicates (or exact duplicates)**: Two job advertisements are considered as full duplicates if they are both exactly the same, i.e. they have the same job title and job description. They may have differing sources and retrieval dates;
 - **Semantic duplicates**: Two job advertisements are considered as semantic duplicates if they advertise the same job position and include the same content in terms of the job characteristics (e.g. the same occupation, the same education or qualification requirements, etc.), but are expressed differently in natural language or in different languages;
- **Temporal duplicates**: Temporal duplicates are semantic duplicates with varying advertisement retrieval dates;
- **Partial duplicates**: Two job advertisements are considered as partial duplicates if they describe the same job position but do not necessarily contain the same characteristics (e.g. one job advertisement contains characteristics that the other does not).

If specific job advertisements cannot be described as full duplicates, partial duplicates, semantic duplicates or temporal duplicates, they are considered non-duplicates.


### Structure
The repository is structured as follows: 

 - [`data`](data/) contains all input, intermediate and output data
 - [`src`](src/) contains all the source code
 - [`tests`](tests/) contains the unit tests 
 - [`main.ipynb`](main.ipynb) is a wrapper jupyter notebook to call all functions and build the final data set
 - [`requirements.txt`](requirements.txt) contains the requirements for the project


### Authors
[Jannic Cutura](https://github.com/JannicCutura)

[Dimitrios Petridis](https://github.com/dim10P)

[Stefan Pasch](https://github.com/Stefan-Pasch)

[Charis Lagonidis](https://github.com/charlago)