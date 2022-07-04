# iswc-challenge

Please use Python 3.10

## Tasks overview

### Prompt creation

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
- Make changes that the competition organisers suggest [priority]
  - Pull the changes from their repo
  - Check our performance on the updated train/val dataset
- Look at failure cases 
  - Wrong formatting?
- Dataset statistics (nice to include in the paper)
  - The number of answers per relation
  - Count the number of 'None' per relation
- Improve precision via logic integrity
    - personEmployer
- Improve recall via
  - Reduce temperature and generate multiple samples (k=3?)
  - Rephrase prompts?
- General improvements 
  - Can we use the logprob?
  - Are we using other models?
