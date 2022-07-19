# Experimental Setup

Thw following are the steps we took to perform our experiments. We report the results on the `dev` dataset.

## Baseline

```bash
python baseline.py -i "data/dev.jsonl" -o "predictions/baseline.pred.jsonl"
python evaluate.py -p "predictions/baseline.pred.jsonl" -g "data/dev.jsonl"
```

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

## Natural Language vs Triple

```bash
#[MAKE ALL PROMPTS WITH EMPTY LISTS]
python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-triple-empty).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002" --simple True
python evaluate.py -p "predictions/gpt3(davinci-dev-triple-empty).pred.jsonl" -g "data/dev.jsonl"

python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-language-empty).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002"
python evaluate.py -p "predictions/gpt3(davinci-dev-language-empty).pred.jsonl" -g "data/dev.jsonl"
```

#### predictions/gpt3(davinci-dev-triple-empty).pred.jsonl

| Relation                  | p     |r   |  f1|
|---------------------------|:------|:------|:------|
| ChemicalCompoundElement   |0.976  |0.919  |0.940|
| CompanyParentOrganization |0.627  |0.640  |0.630|
| CountryBordersWithCountry |0.865  |0.733  |0.766|
| CountryOfficialLanguage   |0.933  |0.810  |0.833|
| PersonCauseOfDeath        |0.560  |0.550  |0.553|
| PersonEmployer            |0.261  |0.267  |0.226|
| PersonInstrument          |0.547  |0.508  |0.502|
| PersonLanguage            |0.840  |0.894  |0.827|
| PersonPlaceOfDeath        |0.820  |0.820  |0.820|
| PersonProfession          |0.669  |0.527  |0.556|
| RiverBasinsCountry        |0.845  |0.868  |0.832|
| StateSharesBorderState    |0.587  |0.407  |0.472|
| *** Average ***           |0.711  |0.662  |0.663|

#### predictions/gpt3(davinci-dev-language-empty).pred.jsonl

| Relation                                    | p     |r   |  f1|
|---------------------------------------------|:------|:------|:------|
| ChemicalCompoundElement  (worse)            |0.895  |0.885  |0.884|
| CompanyParentOrganization (worse)           |0.385  |0.400  |0.388|
| CountryBordersWithCountry  (better)         |0.809  |0.800  |0.785|
| CountryOfficialLanguage   (worse)           |0.798  |0.882  |0.785|
| PersonCauseOfDeath    (worse)           |0.500  |0.500  |0.500|
| PersonEmployer            (better)          |0.273  |0.323  |0.262|
| PersonInstrument             (worse)        |0.489  |0.458  |0.446|
| PersonLanguage             (worse)          |0.750  |0.932  |0.793|
| PersonPlaceOfDeath          (better)        |0.840  |0.840  |0.840|
| PersonProfession            (better)    |0.713  |0.535  |0.581|
| RiverBasinsCountry              (worse) |0.820  |0.863  |0.822|
| StateSharesBorderState        (better) |0.628  |0.462  |0.522|
| *** Average ***                        |0.658  |0.657  |0.634|

## Empty vs None experiment

```bash
#python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-triple-empty).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002" 
#python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-language-empty).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002" 

#[MAKE ALL PROMPTS WITH NONE]
python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-triple-none).pred.jsonl" -i "data/dev(None).jsonl" -m "text-davinci-002"
#[MERGE WITH NON_AFFECTED PREDICTIONS]
python evaluate.py -p "predictions/gpt3(davinci-dev-triple-none).pred.jsonl" -g "data/dev.jsonl"

python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-language-none).pred.jsonl" -i "data/dev(None).jsonl" -m "text-davinci-002"
#[MERGE WITH NON_AFFECTED PREDICTIONS]
python evaluate.py -p "predictions/gpt3(davinci-dev-language-none).pred.jsonl" -g "data/dev.jsonl"
```

#### predictions/gpt3(davinci-dev-triple-none).pred.jsonl

| Relation | p     |r   |  f1|
| ----------------------- |:------|:------|:------|
|ChemicalCompoundElement    |0.976  |0.919  |0.940|
|CompanyParentOrganization  |0.627  |0.640  |0.630|
|CountryBordersWithCountry  |0.826  |0.719  |0.749|
|CountryOfficialLanguage    |0.933  |0.810  |0.833|
|PersonCauseOfDeath         |0.600  |0.590  |0.593|
|PersonEmployer             |0.261  |0.267  |0.226|
|PersonInstrument           |0.600  |0.561  |0.568|
|PersonLanguage             |0.840  |0.894  |0.827|
|PersonPlaceOfDeath         |0.780  |0.780  |0.780|
|PersonProfession           |0.669  |0.527  |0.556|
|RiverBasinsCountry         |0.845  |0.868  |0.832|
|StateSharesBorderState     |0.587  |0.407  |0.472|
|*** Average ***            |0.712  |0.665  |0.667|

#### predictions/gpt3(davinci-dev-language-none).pred.jsonl

| Relation                             | p     |r   |  f1|
|--------------------------------------|:------|:------|:------|
|ChemicalCompoundElement    |0.895  |0.885  |0.884|
|CompanyParentOrganization  |0.385  |0.400  |0.388|
|CountryBordersWithCountry  |0.775  |0.788  |0.758|
|CountryOfficialLanguage    |0.798  |0.882  |0.785|
|PersonCauseOfDeath         |0.560  |0.560  |0.560|
|PersonEmployer             |0.273  |0.323  |0.262|
|PersonInstrument           |0.549  |0.528  |0.522|
|PersonLanguage             |0.750  |0.932  |0.793|
|PersonPlaceOfDeath         |0.780  |0.780  |0.780|
|PersonProfession           |0.713  |0.535  |0.581|
|RiverBasinsCountry         |0.820  |0.863  |0.822|
|StateSharesBorderState     |0.628  |0.462  |0.522|
|*** Average ***            |0.660  |0.661  |0.638|

## [TO REDO] Language Model Size - Scaling

```bash
#[MAKE ALL PROMPTS OPTIMAL ACCORDING TO RESULTS IN GPT3 AND OPT SCRIPTS]
python gpt3_baseline.py -o "predictions/gpt3(ada-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-ada-001" 
python gpt3_baseline.py -o "predictions/gpt3(babbage-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-babbage-001" 
python gpt3_baseline.py -o "predictions/gpt3(curie-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-curie-001" 
python gpt3_baseline.py -o "predictions/gpt3(davinci-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "text-davinci-002" 
#[ASK JAN TO RUN IN SERVER?]
python opt_baseline.py -o "predictions/opt(1.3-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-1.3b"
python opt_baseline.py -o "predictions/opt(6.7-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-6.7b"
python opt_baseline.py -o "predictions/opt(13-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-13b"
python opt_baseline.py -o "predictions/opt(30-dev-language-optimized).pred.jsonl" -i "data/dev.jsonl" -m "facebook/opt-30b"
```

#### predictions/gpt3(ada-dev).pred.jsonl

| Relation                             | p     |r   |  f1|
|--------------------------------------|:------|:------|:------|
|ChemicalCompoundElement    |0.256  |0.225  |0.231|
|CompanyParentOrganization  |0.120  |0.120  |0.120|
|CountryBordersWithCountry  |0.066  |0.040  |0.046|
|CountryOfficialLanguage    |0.142  |0.145  |0.133|
|PersonCauseOfDeath         |0.160  |0.160  |0.160|
|PersonEmployer             |0.000  |0.000  |0.000|
|PersonInstrument           |0.297  |0.352  |0.270|
|PersonLanguage             |0.331  |0.702  |0.394|
|PersonPlaceOfDeath         |0.040  |0.040  |0.040|
|PersonProfession           |0.281  |0.134  |0.156|
|RiverBasinsCountry         |0.365  |0.349  |0.313|
|StateSharesBorderState     |0.102  |0.060  |0.066|
|*** Average ***            |0.180  |0.194  |0.161|

#### predictions/gpt3(babbage-dev).pred.jsonl

| Relation                             | p     |r   |  f1|
|--------------------------------------|:------|:------|:------|
|ChemicalCompoundElement    |0.357  |0.240  |0.275|
|CompanyParentOrganization  |0.080  |0.080  |0.080|
|CountryBordersWithCountry  |0.206  |0.171  |0.170|
|CountryOfficialLanguage    |0.686  |0.629  |0.605|
|PersonCauseOfDeath         |0.040  |0.040  |0.040|
|PersonEmployer             |0.012  |0.017  |0.014|
|PersonInstrument           |0.507  |0.463  |0.457|
|PersonLanguage             |0.689  |0.657  |0.636|
|PersonPlaceOfDeath         |0.000  |0.000  |0.000|
|PersonProfession           |0.513  |0.219  |0.286|
|RiverBasinsCountry         |0.700  |0.558  |0.578|
|StateSharesBorderState     |0.117  |0.078  |0.088|
|*** Average ***            |0.325  |0.263  |0.269|

#### predictions/gpt3(curie-dev).pred.jsonl

| Relation                             | p     |r   |  f1|
|--------------------------------------|:------|:------|:------|
|ChemicalCompoundElement    |0.532  |0.521  |0.513|
|CompanyParentOrganization  |0.140  |0.140  |0.140|
|CountryBordersWithCountry  |0.517  |0.487  |0.462|
|CountryOfficialLanguage    |0.658  |0.768  |0.664|
|PersonCauseOfDeath         |0.040  |0.040  |0.040|
|PersonEmployer             |0.043  |0.067  |0.050|
|PersonInstrument           |0.318  |0.421  |0.326|
|PersonLanguage             |0.752  |0.833  |0.739|
|PersonPlaceOfDeath         |0.000  |0.000  |0.000|
|PersonProfession           |0.683  |0.312  |0.383|
|RiverBasinsCountry         |0.598  |0.711  |0.604|
|StateSharesBorderState     |0.255  |0.195  |0.198|
|*** Average ***            |0.378  |0.375  |0.343|

#### predictions/gpt3(davinci-dev).pred.jsonl

| Relation                             | p     |r   |  f1|
|--------------------------------------|:------|:------|:------|
|ChemicalCompoundElement    |0.905  |0.894  |0.894|
|CompanyParentOrganization  |0.685  |0.700  |0.688|
|CountryBordersWithCountry  |0.830  |0.794  |0.792|
|CountryOfficialLanguage    |0.824  |0.840  |0.788|
|PersonCauseOfDeath         |0.600  |0.590  |0.593|
|PersonEmployer             |0.276  |0.335  |0.270|
|PersonInstrument           |0.589  |0.570  |0.551|
|PersonLanguage             |0.755  |0.936  |0.797|
|PersonPlaceOfDeath         |0.820  |0.820  |0.820|
|PersonProfession           |0.735  |0.526  |0.582|
|RiverBasinsCountry         |0.824  |0.851  |0.820|
|StateSharesBorderState     |0.638  |0.472  |0.532|
|*** Average ***            |0.707  |0.694  |0.677|

## Fact Checking

```bash
python integrity_checking.py -o "predictions/gpt3(davinci-dev-language-optimized)_factcheck.pred.jsonl" -i "predictions/gpt3(davinci-dev-language-optimized).pred.jsonl"
```

## Alias Fetcher

```bash
python wikidata_cleanup.py -o "predictions/gpt3(davinci-dev-language-optimized)_factcheck_wikiclean.pred.jsonl" -i "predictions/gpt3(davinci-dev-language-optimized)_factcheck.pred.jsonl"
```

