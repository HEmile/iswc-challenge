# Current results on dev dataset

### Baseline

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

## None vs empty list

### predictions/gpt3(davinci-dev-empty).pred.jsonl

| Relation | p     |r   |  f1|
| ----------------------- |:------|:------|:------|
|ChemicalCompoundElement    |0.905  |0.894  |0.894|
|CompanyParentOrganization  |0.485  |0.500  |0.488|
|CountryBordersWithCountry  |0.830  |0.794  |0.792|
|CountryOfficialLanguage    |0.824  |0.840  |0.788|
|PersonCauseOfDeath         |0.560  |0.560  |0.560|
|PersonEmployer             |0.270  |0.333  |0.266|
|PersonInstrument           |0.589  |0.570  |0.551|
|PersonLanguage             |0.759  |0.941  |0.801|
|PersonPlaceOfDeath         |0.820  |0.820  |0.820|
|PersonProfession           |0.735  |0.526  |0.582|
|RiverBasinsCountry         |0.824  |0.846  |0.817|
|StateSharesBorderState     |0.621  |0.463  |0.519|
|*** Average ***            |0.685  |0.674  |0.657|

### gpt3(davinci-dev-None).pred.jsonl

| Relation                             | p     |r   |  f1|
|--------------------------------------|:------|:------|:------|
| ChemicalCompoundElement              | 0.905  | 0.894  | 0.894|
| CompanyParentOrganization   (better) | 0.685  | 0.700  | 0.688|
| CountryBordersWithCountry (worse)    | 0.815  | 0.782  | 0.778|
| CountryOfficialLanguage              | 0.824  | 0.840  | 0.788|
| PersonCauseOfDeath      (better)     | 0.580  | 0.580  | 0.580|
| PersonEmployer                       | 0.270  | 0.333  | 0.266|
| PersonInstrument     (worse)         | 0.545  | 0.520  | 0.519|
| PersonLanguage                       | 0.759  | 0.941  | 0.801|
| PersonPlaceOfDeath     (worse)       | 0.800  | 0.800  | 0.800|
| PersonProfession                     | 0.735  | 0.526  | 0.582|
| RiverBasinsCountry                   | 0.824  | 0.846  | 0.817|
| StateSharesBorderState               | 0.621  | 0.463  | 0.519|
| *** Average ***                      | 0.697  | 0.685  | 0.669|

## Scaling

### predictions/gpt3(ada-dev).pred.jsonl

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

### predictions/gpt3(babbage-dev).pred.jsonl 

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

### predictions/gpt3(curie-dev).pred.jsonl 

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

### predictions/gpt3(davinci-dev).pred.jsonl 

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

## Fact probing

### GPT3 (['None']) + fact checking

| Relation  | p     |r   |  f1|
|--------------------------|:------|:------|:------|
|ChemicalCompoundElement    | 0.902  | 0.891  | 0.890|
|CompanyParentOrganization  | 0.640  | 0.640  | 0.640|
|CountryBordersWithCountry  | 0.830  | 0.794  | 0.792|
|CountryOfficialLanguage    | 0.876  | 0.810  | 0.794|
|PersonCauseOfDeath         | 0.580  | 0.580  | 0.580|
|PersonEmployer             | 0.270  | 0.333  | 0.266|
|PersonInstrument           | 0.663  | 0.612  | 0.617|
|PersonLanguage             | 0.863  | 0.849  | 0.825|
|PersonPlaceOfDeath         | 0.820  | 0.820  | 0.820|
|PersonProfession           | 0.725  | 0.520  | 0.574|
|RiverBasinsCountry         | 0.824  | 0.851  | 0.820|
|StateSharesBorderState     | 0.622  | 0.464  | 0.521|
|*** Average ***            | 0.718  | 0.680  | 0.678|

