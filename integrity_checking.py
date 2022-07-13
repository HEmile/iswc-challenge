
# Get positive and negative fact queries
# See with what probabilities it predicts true or false
import argparse
import json
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import time

from pandas import DataFrame

from utils.file_io import read_lm_kbc_jsonl_to_df, df_to_jsonl
from utils.model import gpt3, clean_up


def logical_integrity(batch: pd.DataFrame) -> List[Tuple[Tuple[int, str], pd.DataFrame]]:
    prompts = []
    indices = []
    for index, subject, relation, _, objects in batch.itertuples(index=True):
        for object in objects:
            prompts.append(positive_negative_prompt_pairs(relation, subject, object))
            indices.append((index, object))
    predictions = []
    for ndx in range(0, len(prompts), 20):
        predictions.extend(gpt3(prompts[ndx:min(ndx+20, len(prompts))]))
        time.sleep(2)
    # predictions = gpt3(prompts)
    # for i, prediction in enumerate(predictions):
    #     print(prompts[i])
    #     print(prediction['text'])
    #     print("\n")
    return list(zip(indices, predictions))

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

{subject_entity} is a {object_entity.lower()}.
"""

    elif relation == "PersonInstrument":
        prompt = f"""
Liam Gallagher plays the guitar.
True

Jay Park plays the piano.
False        
        
{subject_entity} plays the {object_entity.lower()}.
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


def fact_checking(input_file, output_file):
    ### looping over all the files in the input directory
    prompt_df = read_lm_kbc_jsonl_to_df(input_file)
    filtered = logical_integrity(prompt_df)
    indices = []
    for index, prediction in filtered:
        if prediction['text'] == 'False':
            indices.append(index)
    for index, object in indices:
        prompt_df["ObjectEntities"][index].remove(object)
    with open(output_file, "w") as f:
        for prediction in df_to_jsonl(prompt_df):
            f.write(json.dumps(prediction) + "\n")

    # TODO: If by filtering a fact, there are no more objects for a certain subject, make sure to add NONE



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="./predictions/gpt3.pred.jsonl",
        help="input directory containing the baseline or your method output",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="./predictions/gpt3_fact_check.pred.jsonl",
        help="Output file (required)",
    )

    args = parser.parse_args()
    print(args)

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    assert input_file.exists()

    fact_checking(input_file, output_file)

if __name__ == "__main__":
    main()