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

### GPT3 ([]) - OLD RUN

| Relation | p     |r   |  f1|
| ----------------------- |:------|:------|:------|
|**ChemicalCompoundElement**  | 0.902 |0.891  |0.890 | 
|CompanyParentOrganization | 0.485 |0.500  |0.488 |
|CountryBordersWithCountry | 0.830 |0.794  |0.792 |
|**CountryOfficialLanguage**  | 0.834 |0.843  |0.793 |
|PersonCauseOfDeath       | 0.560 |0.560  |0.560 |
|**PersonEmployer**           | 0.266 |0.315  |0.256 |
|**PersonInstrument**         | 0.589 |0.570  |0.551 |
|**PersonLanguage**           | 0.762 |0.936  |0.801 |
|PersonPlaceOfDeath       | 0.820 |0.820  |0.820 |
|**PersonProfession**         | 0.720 |0.513  |0.569 |
|**RiverBasinsCountry**       | 0.824 |0.851  |0.820 |
|**StateSharesBorderState**   | 0.636 |0.474  |0.532 |
|***Average***            | 0.686 |0.672  |0.656 |

### GPT3 (['None'])

| Relation  | p     |r   |  f1|
|--------------------------|:------|:------|:------|
| ChemicalCompoundElement    | 0.905  | 0.894  | 0.894 |
| CompanyParentOrganization  | 0.485  | 0.500  | 0.488 |
| CountryBordersWithCountry  | 0.830  | 0.794  | 0.792 |
| CountryOfficialLanguage    | 0.824  | 0.840  | 0.788 |
| PersonCauseOfDeath         | 0.560  | 0.560  | 0.560 |
| PersonEmployer             | 0.276  | 0.335  | 0.270 |
| PersonInstrument           | 0.569  | 0.550  | 0.531 |
| PersonLanguage             | 0.755  | 0.936  | 0.797 |
| PersonPlaceOfDeath         | 0.820  | 0.820  | 0.820 |
| PersonProfession           | 0.718  | 0.515  | 0.569 |
| RiverBasinsCountry         | 0.824  | 0.851  | 0.820 |
| StateSharesBorderState     | 0.643  | 0.481  | 0.539 |
| *** Average ***            | 0.684  | 0.673  | 0.656 |

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

