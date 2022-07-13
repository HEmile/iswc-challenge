# iswc-challenge

## Getting started

### Prerequisites

This repository uses Python >= 3.10

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. In the root directory of this repo run

    ```bash
    pip install requirements.txt
    ```

### Usage

For running and evaluating the baseline, run :

```bash
python baseline.py -i "data/dev.jsonl" -o "predictions/baseline.pred.jsonl"
python evaluate.py -p "predictions/baseline.pred.jsonl" -g "data/dev.jsonl"
```

For running and evaluating our proposed GPT3 approach, make sure you set your `OPENAI_API_KEY` in the environmental
variables. Then, run :

```bash
python gpt3_baseline
python evaluate.py -p "predictions/gpt3.pred.jsonl" -g "data/dev.jsonl"
```

## Current results

* Baseline

| Relation | p     |r   |  f1|
| ----------------------- |:------|:------|:------|
|ChemicalCompoundElement  | 0.140  | 0.060  | 0.083|
|CompanyParentOrganization | 0.680  | 0.680  | 0.680|
|CountryBordersWithCountry | 0.255  | 0.087  | 0.122|
|CountryOfficialLanguage  | 0.894  | 0.703  | 0.752|
|PersonCauseOfDeath       | 0.420  | 0.420  | 0.420|
|PersonEmployer           | 0.000  | 0.000  | 0.000|
|PersonInstrument         | 0.340  | 0.340  | 0.340|
|PersonLanguage           | 0.480  | 0.412  | 0.431|
|PersonPlaceOfDeath       | 0.500  | 0.500  | 0.500|
|PersonProfession         | 0.000  | 0.000  | 0.000|
|RiverBasinsCountry       | 0.480  | 0.342  | 0.381|
|StateSharesBorderState   | 0.000  | 0.000  | 0.000|
|***Average***            | 0.349  | 0.295  | 0.309|

* GPT3

| Relation | p     |r   |  f1|
| ----------------------- |:------|:------|:------|
|ChemicalCompoundElement  | 0.902 |0.891  |0.890 |
|CompanyParentOrganization | 0.485 |0.500  |0.488 |
|CountryBordersWithCountry | 0.830 |0.794  |0.792 |
|CountryOfficialLanguage  | 0.834 |0.843  |0.793 |
|PersonCauseOfDeath       | 0.560 |0.560  |0.560 |
|PersonEmployer           | 0.266 |0.315  |0.256 |
|PersonInstrument         | 0.589 |0.570  |0.551 |
|PersonLanguage           | 0.762 |0.936  |0.801 |
|PersonPlaceOfDeath       | 0.820 |0.820  |0.820 |
|PersonProfession         | 0.720 |0.513  |0.569 |
|RiverBasinsCountry       | 0.824 |0.851  |0.820 |
|StateSharesBorderState   | 0.636 |0.474  |0.532 |
|***Average***            | 0.686 |0.672  |0.656 |

## Tasks overview

### GPT-3 prompt creation

| Relation | Main person in charge | GPT-3 |
| ------------------------- |:----------------------|:------|
| CountryBordersWithCountry | Selene                | DONE  |
| CountryOfficialLanguage | Emile                 | DONE |
| RiverBasinsCountry | Emile                 | DONE |
|StateSharesBorderState | Emile                 | DONE  |
|ChemicalCompoundElement | Thiviyan              | DONE  |
|PersonInstrument | Emile                 | DONE  |
|PersonLanguage | Dimitris              | DONE  |
|PersonEmployer | Thiviyan              | DONE |
|PersonProfession | Dimitris              | DONE  |
|PersonPlaceOfDeath | Dimitris              | DONE  |
|PersonCauseOfDeath | Dimitris              | DONE  |
|CompanyParentOrganization | Thiviyan              | DONE |

## Improvements:

- [X] Make changes that the competition organisers suggest [priority]
    - [X] Pull the changes from their repo
    - [X] Check our performance on the updated train/val dataset
- [X] Dataset statistics (nice to include in the paper)
    - [X] The number of answers per relation
    - [X] Count the number of 'None' per relation
- [X] Improve precision via logic integrity
    - [ ] personEmployer
- Submit current version to leadership board
- [ ] Look at failure cases
    - [ ] Wrong formatting?
- [ ] Improve recall via
    - [ ] Reduce temperature and generate multiple samples (k=3?)
    - [ ] Rephrase prompts?
- [ ] General improvements
    - [ ] Can we use the logprob?
    - [ ] Are we using other models?


## License

Distributed under the MIT License.
See [`LICENSE`]() for more information.

## Authors

* [Selene Báez Santamaría](https://selbaez.github.io/)
