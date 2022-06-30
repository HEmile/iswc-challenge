
# Get positive and negative fact queries
# See with what probabilities it predicts true or false
from typing import List


def logical_integrity(results: List[dict]):
    pass

def positive_negative_prompt_pairs(relation, subject_entity, object_entity):
    ### depending on the relation, we fix the prompt
    if relation == "CountryBordersWithCountry":
        prompt = f"""
Niger neighbours Libya.
True

North Korea neighbours the Netherlands.
False

{subject_entity} neighbours {object_entity}.
"""
    elif relation == "CountryOfficialLanguage":
        prompt = f"""
Swedish is the official language of Finland.
True

French is the official language of India.
False

{object_entity} is the official language of {subject_entity}.
"""
    elif relation == "StateSharesBorderState":
        prompt = f"""
San Marino shares a border with San Leo.
True

Texas shares a border with Hamburg.
False

{subject_entity} shares a border with {object_entity}.
"""
    elif relation == "RiverBasinsCountry":
        prompt = f"""
The river Drava crosses Hungary.
True

The river Huai crosses the Netherlands.
False

The river {subject_entity} crosses {object_entity}.
"""

    elif relation == "ChemicalCompoundElement":
        prompt = f"""
The molecule water is made up of the element Hydrogen.
True

The molecule aspirin is made up of the element Germanium.
False
        
The molecule {subject_entity} is made up of the element {object_entity}.
"""
    elif relation == "PersonLanguage":
        prompt = f"""
Aamir Khan speaks Hindi.
True

Pharrell Williams speaks French.
False

{subject_entity} speaks {object_entity}.
"""

    elif relation == "PersonProfession":
        prompt = f"""
Danny DeVito is a director.
True

Christina Aguilera is a businessperson.
False

{subject_entity} is a {object_entity}.
"""

    elif relation == "PersonInstrument":
        prompt = f"""
Liam Gallagher plays the guitar.
True

Jay Park plays the piano.
False        
        
{subject_entity} plays the {object_entity}.
"""
    elif relation == "PersonEmployer":
        prompt = f"""
Susan Wojcicki is or was employed by Google.
True

Steve Wozniak is or was employed by Microsoft.
False

{subject_entity} is or was employed by {object_entity}.
"""
    elif relation == "PersonPlaceOfDeath":
        prompt = f"""
The place of death of Elvis Presley is Graceland.
True

The place of death of Barack Obama is Washington.
False

The place of death of {subject_entity} is {object_entity}.
"""

    elif relation == "PersonCauseOfDeath":
        prompt = f"""
Aretha Franklin died of pancreatic cancer.
True

Bill Gates died of femoral fracture.
False

{subject_entity} died of {object_entity}. 
"""

    elif relation == "CompanyParentOrganization":
        prompt = f"""
Apple is the parent company of Microsoft.
False

Sony Group is the parent company of Sony.
True

{object_entity} is the parent company of {subject_entity}?
"""
    return prompt

