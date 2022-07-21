# Experimental Setup

Thw following are the steps we took to perform our experiments. We report the results on the `dev` dataset.

## Baseline

```bash
python baseline.py -i "data/dev.jsonl" -o "predictions/baseline.pred.jsonl"
python evaluate.py -p "predictions/baseline.pred.jsonl" -g "data/dev.jsonl"
```

| Relation                  | p     | r     | f1    |
| ------------------------- | :---- | :---- | :---- |
| ChemicalCompoundElement   | 0.140 | 0.060 | 0.083 |
| CompanyParentOrganization | 0.680 | 0.680 | 0.680 |
| CountryBordersWithCountry | 0.255 | 0.087 | 0.122 |
| CountryOfficialLanguage   | 0.894 | 0.703 | 0.752 |
| PersonCauseOfDeath        | 0.420 | 0.420 | 0.420 |
| PersonEmployer            | 0.000 | 0.000 | 0.000 |
| PersonInstrument          | 0.340 | 0.340 | 0.340 |
| PersonLanguage            | 0.480 | 0.412 | 0.431 |
| PersonPlaceOfDeath        | 0.500 | 0.500 | 0.500 |
| PersonProfession          | 0.000 | 0.000 | 0.000 |
| RiverBasinsCountry        | 0.480 | 0.342 | 0.381 |
| StateSharesBorderState    | 0.000 | 0.000 | 0.000 |
| ***Average***             | 0.349 | 0.295 | 0.309 |

## Natural Language vs Triple

```bash
#[MAKE ALL PROMPTS WITH EMPTY LISTS]
python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-triple-empty).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002" --prompt_type 'triple'
python evaluate.py -p "predictions/gpt3(davinci-dev-triple-empty).pred.jsonl" -g "data/dev.jsonl"

python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-language-empty).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002" --prompt_type 'language'
python evaluate.py -p "predictions/gpt3(davinci-dev-language-empty).pred.jsonl" -g "data/dev.jsonl"
```

#### predictions/gpt3(davinci-dev-triple-empty).pred.jsonl

| Relation                  | p     | r     | f1    |
| ------------------------- |:------|:------|:------|
| ChemicalCompoundElement   | 0.976 | 0.919 | 0.940 |
| CompanyParentOrganization | 0.587 | 0.600 | 0.590 |
| CountryBordersWithCountry | 0.865 | 0.733 | 0.766 |
| CountryOfficialLanguage   | 0.933 | 0.810 | 0.833 |
| PersonCauseOfDeath        | 0.560 | 0.550 | 0.553 |
| PersonEmployer            | 0.261 | 0.267 | 0.226 |
| PersonInstrument          | 0.547 | 0.508 | 0.502 |
| PersonLanguage            | 0.840 | 0.894 | 0.827 |
| PersonPlaceOfDeath        | 0.820 | 0.820 | 0.820 |
| PersonProfession          | 0.669 | 0.527 | 0.556 |
| RiverBasinsCountry        | 0.845 | 0.868 | 0.832 |
| StateSharesBorderState    | 0.587 | 0.407 | 0.472 |
| *** Average ***           | 0.707 | 0.658 | 0.660 |

#### predictions/gpt3(davinci-dev-language-empty).pred.jsonl

| Relation                  | p     | r     | f1    |
| ------------------------- | :---- | :---- | :---- |
| ChemicalCompoundElement   | 0.895 | 0.885 | 0.884 |
| CompanyParentOrganization | 0.385 | 0.400 | 0.388 |
| CountryBordersWithCountry | 0.809 | 0.800 | 0.785 |
| CountryOfficialLanguage   | 0.798 | 0.882 | 0.785 |
| PersonCauseOfDeath        | 0.500 | 0.500 | 0.500 |
| PersonEmployer            | 0.273 | 0.323 | 0.262 |
| PersonInstrument          | 0.489 | 0.458 | 0.446 |
| PersonLanguage            | 0.750 | 0.932 | 0.793 |
| PersonPlaceOfDeath        | 0.840 | 0.840 | 0.840 |
| PersonProfession          | 0.713 | 0.535 | 0.581 |
| RiverBasinsCountry        | 0.820 | 0.863 | 0.822 |
| StateSharesBorderState    | 0.628 | 0.462 | 0.522 |
| *** Average ***           | 0.658 | 0.657 | 0.634 |

## Empty vs None experiment

```bash
#[MAKE ALL PROMPTS WITH NONE]
python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-triple-none).pred.jsonl" -i "data/dev(None).jsonl" -m "text-davinci-002" --prompt_type 'triple'
#[MERGE WITH NON_AFFECTED PREDICTIONS]
python evaluate.py -p "predictions/gpt3(davinci-dev-triple-none).pred.jsonl" -g "data/dev.jsonl"

python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-language-none).pred.jsonl" -i "data/dev(None).jsonl" -m "text-davinci-002" --prompt_type 'language'
#[MERGE WITH NON_AFFECTED PREDICTIONS]
python evaluate.py -p "predictions/gpt3(davinci-dev-language-none).pred.jsonl" -g "data/dev.jsonl"
```

#### predictions/gpt3(davinci-dev-triple-none).pred.jsonl

| Relation                  | p     | r     | f1    |
| ------------------------- |:------|:------|:------|
| ChemicalCompoundElement   | 0.976 | 0.919 | 0.940 |
| CompanyParentOrganization | 0.767 | 0.780 | 0.770 |
| CountryBordersWithCountry | 0.826 | 0.719 | 0.749 |
| CountryOfficialLanguage   | 0.933 | 0.810 | 0.833 |
| PersonCauseOfDeath        | 0.600 | 0.590 | 0.593 |
| PersonEmployer            | 0.261 | 0.267 | 0.226 |
| PersonInstrument          | 0.600 | 0.561 | 0.568 |
| PersonLanguage            | 0.840 | 0.894 | 0.827 |
| PersonPlaceOfDeath        | 0.780 | 0.780 | 0.780 |
| PersonProfession          | 0.669 | 0.527 | 0.556 |
| RiverBasinsCountry        | 0.845 | 0.868 | 0.832 |
| StateSharesBorderState    | 0.587 | 0.407 | 0.472 |
| *** Average ***           | 0.724 | 0.677 | 0.679 |

#### predictions/gpt3(davinci-dev-language-none).pred.jsonl

| Relation                  | p     | r     | f1    |
| ------------------------- | :---- | :---- | :---- |
| ChemicalCompoundElement   | 0.895 | 0.885 | 0.884 |
| CompanyParentOrganization | 0.385 | 0.400 | 0.388 |
| CountryBordersWithCountry | 0.775 | 0.788 | 0.758 |
| CountryOfficialLanguage   | 0.798 | 0.882 | 0.785 |
| PersonCauseOfDeath        | 0.560 | 0.560 | 0.560 |
| PersonEmployer            | 0.273 | 0.323 | 0.262 |
| PersonInstrument          | 0.549 | 0.528 | 0.522 |
| PersonLanguage            | 0.750 | 0.932 | 0.793 |
| PersonPlaceOfDeath        | 0.780 | 0.780 | 0.780 |
| PersonProfession          | 0.713 | 0.535 | 0.581 |
| RiverBasinsCountry        | 0.820 | 0.863 | 0.822 |
| StateSharesBorderState    | 0.628 | 0.462 | 0.522 |
| *** Average ***           | 0.660 | 0.661 | 0.638 |

### Final decision per relation type

| Relation                  | Prompt type<br/> |
| ------------------------- | :--------------- |
| ChemicalCompoundElement   | triple           |
| CompanyParentOrganization | triple, none     |
| CountryBordersWithCountry | language, empty  |
| CountryOfficialLanguage   | triple           |
| PersonCauseOfDeath        | triple, none     |
| PersonEmployer            | language         |
| PersonInstrument          | triple, none     |
| PersonLanguage            | triple           |
| PersonPlaceOfDeath        | language, empty  |
| PersonProfession          | language         |
| RiverBasinsCountry        | triple           |
| StateSharesBorderState    | language         |

## Language Model Size - Scaling

```bash
#[MAKE ALL PROMPTS OPTIMAL ACCORDING TO RESULTS - GPT3 AND OPT SCRIPTS]
python gpt3_baseline.py -o "predictions/gpt3(ada-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-ada-001" --prompt_type 'optimized'
python gpt3_baseline.py -o "predictions/gpt3(babbage-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-babbage-001"  --prompt_type 'optimized'
python gpt3_baseline.py -o "predictions/gpt3(curie-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-curie-001"  --prompt_type 'optimized'
python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002"  --prompt_type 'optimized'
#[ASK JAN TO RUN IN SERVER]
python opt_baseline.py -o "predictions/opt(1.3-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-1.3b"
python opt_baseline.py -o "predictions/opt(6.7-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-6.7b"
python opt_baseline.py -o "predictions/opt(13-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-13b"
python opt_baseline.py -o "predictions/opt(30-dev-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-30b"
```

#### [TO REDO] predictions/gpt3(ada-dev).pred.jsonl 

| Relation                  | p     | r     | f1    |
| ------------------------- | :---- | :---- | :---- |
| ChemicalCompoundElement   | 0.256 | 0.225 | 0.231 |
| CompanyParentOrganization | 0.120 | 0.120 | 0.120 |
| CountryBordersWithCountry | 0.066 | 0.040 | 0.046 |
| CountryOfficialLanguage   | 0.142 | 0.145 | 0.133 |
| PersonCauseOfDeath        | 0.160 | 0.160 | 0.160 |
| PersonEmployer            | 0.000 | 0.000 | 0.000 |
| PersonInstrument          | 0.297 | 0.352 | 0.270 |
| PersonLanguage            | 0.331 | 0.702 | 0.394 |
| PersonPlaceOfDeath        | 0.040 | 0.040 | 0.040 |
| PersonProfession          | 0.281 | 0.134 | 0.156 |
| RiverBasinsCountry        | 0.365 | 0.349 | 0.313 |
| StateSharesBorderState    | 0.102 | 0.060 | 0.066 |
| *** Average ***           | 0.180 | 0.194 | 0.161 |

#### [TO REDO] predictions/gpt3(babbage-dev).pred.jsonl

| Relation                  | p     | r     | f1    |
| ------------------------- | :---- | :---- | :---- |
| ChemicalCompoundElement   | 0.357 | 0.240 | 0.275 |
| CompanyParentOrganization | 0.080 | 0.080 | 0.080 |
| CountryBordersWithCountry | 0.206 | 0.171 | 0.170 |
| CountryOfficialLanguage   | 0.686 | 0.629 | 0.605 |
| PersonCauseOfDeath        | 0.040 | 0.040 | 0.040 |
| PersonEmployer            | 0.012 | 0.017 | 0.014 |
| PersonInstrument          | 0.507 | 0.463 | 0.457 |
| PersonLanguage            | 0.689 | 0.657 | 0.636 |
| PersonPlaceOfDeath        | 0.000 | 0.000 | 0.000 |
| PersonProfession          | 0.513 | 0.219 | 0.286 |
| RiverBasinsCountry        | 0.700 | 0.558 | 0.578 |
| StateSharesBorderState    | 0.117 | 0.078 | 0.088 |
| *** Average ***           | 0.325 | 0.263 | 0.269 |

#### [TO REDO]  predictions/gpt3(curie-dev).pred.jsonl

| Relation                  | p     | r     | f1    |
| ------------------------- | :---- | :---- | :---- |
| ChemicalCompoundElement   | 0.532 | 0.521 | 0.513 |
| CompanyParentOrganization | 0.140 | 0.140 | 0.140 |
| CountryBordersWithCountry | 0.517 | 0.487 | 0.462 |
| CountryOfficialLanguage   | 0.658 | 0.768 | 0.664 |
| PersonCauseOfDeath        | 0.040 | 0.040 | 0.040 |
| PersonEmployer            | 0.043 | 0.067 | 0.050 |
| PersonInstrument          | 0.318 | 0.421 | 0.326 |
| PersonLanguage            | 0.752 | 0.833 | 0.739 |
| PersonPlaceOfDeath        | 0.000 | 0.000 | 0.000 |
| PersonProfession          | 0.683 | 0.312 | 0.383 |
| RiverBasinsCountry        | 0.598 | 0.711 | 0.604 |
| StateSharesBorderState    | 0.255 | 0.195 | 0.198 |
| *** Average ***           | 0.378 | 0.375 | 0.343 |

#### predictions/gpt3(davinci-dev).pred.jsonl

| Relation                  | p     | r     | f1   |
|---------------------------|:------|:------|:-----|
|ChemicalCompoundElement    |0.976  | 0.919 | 0.940|
|CompanyParentOrganization  |0.767  | 0.780 | 0.770 |
|CountryBordersWithCountry  |0.809  | 0.800 | 0.785|
|CountryOfficialLanguage    |0.933  | 0.810 | 0.833|
|PersonCauseOfDeath         |0.600  | 0.590 | 0.593|
|PersonEmployer             |0.281  | 0.333 | 0.271|
|PersonInstrument           |0.600  | 0.561 | 0.568|
|PersonLanguage             |0.843  | 0.894 | 0.830|
|PersonPlaceOfDeath         |0.840  | 0.840 | 0.840|
|PersonProfession           |0.709  | 0.536 | 0.581|
|RiverBasinsCountry         |0.845  | 0.868 | 0.832|
|StateSharesBorderState     |0.628  | 0.462 | 0.522 |
|*** Average ***            |0.736  | 0.699 | 0.697 |

## Fact Checking

```bash
python integrity_checking.py -o "predictions/gpt3(davinci-dev-optimized)_factcheck.pred.jsonl" -i "predictions/gpt3(davinci-dev-optimized).pred.jsonl"
```

| Relation                          | p     | r      |  f1|
|---------------------------|:------|:-------|:------|
|ChemicalCompoundElement    |0.976  | 0.919  |0.940|
|CompanyParentOrganization  |0.800  | 0.800  | 0.800 |
|CountryBordersWithCountry  |0.809  | 0.800  |0.785|
|CountryOfficialLanguage    |0.933  | 0.810  |0.833|
|PersonCauseOfDeath         |0.600  | 0.590  |0.593|
|PersonEmployer             |0.281  | 0.333  |0.271|
|PersonInstrument           |0.563  | 0.522  |0.534|
|PersonLanguage             |0.913  | 0.829  |0.842|
|PersonPlaceOfDeath         |0.840  | 0.840  |0.840|
|PersonProfession           |0.709  | 0.536  |0.581|
|RiverBasinsCountry         |0.845  | 0.868  |0.832|
|StateSharesBorderState     |0.628  | 0.462 | 0.522 |
|*** Average ***            |0.741 |  0.692 | 0.698 |

## Alias Fetcher

To run the alias fetcher, first get the index from https://surfdrive.surf.nl/files/index.php/s/SERFiv6ym8OEI9U

Then, run the cleanup using:

```bash
python wikidata_cleanup.py  -i "predictions/gpt3(davinci-dev-optimized)_factcheck.pred.jsonl" -o "predictions/gpt3(davinci-dev-optimized)_factcheck_wikiclean.pred.jsonl"
python evaluate.py -p "predictions/gpt3(davinci-dev-optimized)_factcheck_wikiclean.pred.jsonl" -g "data/dev.jsonl"
```

|                           | p     | r     | f1    |
| ------------------------- |:------|:------|:------|
| ChemicalCompoundElement   | 0.986 | 0.929 | 0.950 |
| CompanyParentOrganization | 0.820 | 0.820 | 0.820 |
| CountryBordersWithCountry | 0.811 | 0.820 | 0.790 |
| CountryOfficialLanguage   | 0.933 | 0.810 | 0.833 |
| PersonCauseOfDeath        | 0.640 | 0.630 | 0.633 |
| PersonEmployer            | 0.336 | 0.418 | 0.336 |
| PersonInstrument          | 0.563 | 0.522 | 0.534 |
| PersonLanguage            | 0.913 | 0.829 | 0.842 |
| PersonPlaceOfDeath        | 0.840 | 0.840 | 0.840 |
| PersonProfession          | 0.712 | 0.539 | 0.584 |
| RiverBasinsCountry        | 0.865 | 0.878 | 0.846 |
| StateSharesBorderState    | 0.621 | 0.430 | 0.499 |
| *** Average ***           | 0.755 | 0.709 | 0.712 |