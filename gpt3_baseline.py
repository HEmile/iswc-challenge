import argparse
import os
from pathlib import Path

import openai
import pandas as pd
import torch

SAMPLE_SIZE = 5

openai.api_key = os.getenv("OPENAI_API_KEY")

### using GPU if available
device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(1000)

RELATIONS = {
    "CountryBordersWithCountry",
    "CountryOfficialLanguage",
    "StateSharesBorderState",
    "RiverBasinsCountry",
    "ChemicalCompoundElement",
    "PersonLanguage",
    "PersonProfession",
    "PersonInstrument",
    "PersonEmployer",
    "PersonPlaceOfDeath",
    "PersonCauseOfDeath",
    "CompanyParentOrganization",
}


def gpt3(prompt):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return response.choices[0]['text'], response.choices[0]['logprobs']['tokens'], response.choices[0]['logprobs']['token_logprobs']


def clean_up(text):
    """ functions to clean up api output """
    return text.strip()


def create_prompt(subject_entity, relation):
    ### depending on the relation, we fix the prompt
    if relation == "CountryBordersWithCountry":
        prompt = f"""
Which countries neighbour Niger?
['Burkina Faso', 'Benin', 'Libya', 'Mali', 'Algeria', 'Chad', 'Nigeria']

Which countries neighbour North Korea?
['South Korea', 'China', 'Russia']

Which countries neighbour Serbia?
['Montenegro', 'Kosovo', 'Bosnia and Herzegovina', 'Hungary', 'Croatia', 'Bulgaria',  'Macedonia', 'Albania', 'Romania']

Which countries neighbour Fiji?
['NONE']

Which countries neighbour {subject_entity}?
"""

    elif relation == "CountryOfficialLanguage":
        prompt = f"""
Which are the official languages of Finland?
['swedish', 'finnish']

Which are the official languages of India?
['english', 'hindi']

Which are the official languages of Norway?
['norwegian', 'nynorsk', 'sami', 'sámi', 'bokmal', 'nynorsk']

Which are the official languages of Granada?
['grenadian creole english', 'english', 'creole', 'grenadian']

Which are the official languages of {subject_entity}?        
"""
    elif relation == "StateSharesBorderState":
        prompt = f"""
What states border San Marino?
['san leo', 'acquaviva', 'borgo maggiore', 'chiesanuova', 'fiorentino']

What states border Texas?
['chihuahua', 'new mexico, 'nuevo león', 'tamaulipas', 'coahuila', 'louisiana', 'arkansas', 'oklahoma']

What states border Liguria?
['tuscany', 'auvergne-rhoone-alpes', 'piedmont', 'emilia-romagna']

What states border Mecklenberg-western pomerania?
['brandenburg', 'pomeranian', 'schleswig-holstein', 'lower saxony']

What states border {subject_entity}?
"""

    elif relation == "RiverBasinsCountry":
        prompt = f"""
Which instruments does Liam Gallagher play?
['maraca', 'guitar']

Which instruments does Liam Gallagher play?
['upright piano', 'piano', 'guitar', 'harmonica']

Which instruments does Jay Park play?
['NONE']

Which instruments does Axl Rose play?
['guitar', 'piano', 'pander', 'bass']

Which instruments does Neil Young play?
['guitar']

Which instruments does {subject_entity} play?
"""

    elif relation == "ChemicalCompoundElement":
        prompt = f"""
What are all the chemical elements that make up the molecule water?
['Hydrogen', 'Water']

What are all the chemical elements that make up the molecule aspirin?
['Oxygen', 'Carbon', 'Hydrogen']

What are all the chemical elements that make up the molecule {subject_entity}?
"""
    elif relation == "PersonLanguage":
        prompt = f"""
Which languages does Aamir Khan speak?
['hindi', 'english', 'urdu']

Which languages does Pharrell Williams speak?
['english']

Which languages does Shakira speak?
['catalan', 'english', 'portuguese', 'spanish']

Which languages does {subject_entity} speak?
"""

    elif relation == "PersonProfession":
        prompt = f"""
What is Danny DeVito's profession?
['director', 'film director'] 

What is Christina Aguilera's profession?
['artist', 'recording artist']

What is Donald Trump's profession?
['businessperson', 'conspiracy theorist', 'politician']

What is {subject_entity}'s profession?
"""

    elif relation == "PersonInstrument":
        prompt = f"""
Which instruments does Liam Gallagher play?
['maraca', 'guitar']

Which instruments does Liam Gallagher play?
['upright piano', 'piano', 'guitar', 'harmonica']

Which instruments does Jay Park play?
['NONE']

Which instruments does Axl Rose play?
['guitar', 'piano', 'pander', 'bass']

Which instruments does Neil Young play?
['guitar']

Which instruments does {subject_entity} play?
"""
    elif relation == "PersonEmployer":
        prompt = f"""
Where is or was Susan Wojcicki employed?
['Google']

Where is or was Steve Wozniak employed?
['Apple Inc', 'Hewlett-Packard', 'University of Technology Sydney', 'Atari, Atari Inc']

Where is or was {subject_entity} employed?
"""
    elif relation == "PersonPlaceOfDeath":
        prompt = f"""
What is the place of death of Barack Obama?
['NONE']

What is the place of death of Ennio morricone?
['rome']

What is the place of death of Elvis presley?
['graceland']

What is the place of death of Elon musk?
['NONE']

What is the place of death of Prince?
['chanhassen']

What is the place of death of {subject_entity}? 
"""

    elif relation == "PersonCauseOfDeath":
        prompt = f"""
How did Aretha Franklin die?
['pancreatic cancer', 'cancer']

How did Bill Gates die?
['NONE']

How did Ennio Morricone die?
['femoral fracture', 'fracture']

How did Frank Sinatra die?
['myocardial infarction', 'infarction']

How did Michelle Obama die?
['NONE']

How did {subject_entity} die? 
"""

    elif relation == "CompanyParentOrganization":
        prompt = f"""
What is the parent company of Microsoft?
['None']

What is the parent company of Sony?
['sony group', 'sony']

What is the parent company of {subject_entity}?
"""
    return prompt


def probe_lm(relation, subject_entities, output_dir: Path):
    ### for every subject-entity in the entities list, we probe the LM using the below sample prompts
    results = []
    for index, subject_entity in enumerate(subject_entities):
        print(f"Probing the GPT3 language model "
              f"for {subject_entity} (subject-entity) and {relation} relation")

        # TODO: Generate examples in the prompt automatically (Thiviyan)

        # TODO: Rephrase prompt automatically (Dimitris)

        ### creating a specific prompt for the given relation
        prompt = create_prompt(subject_entity, relation)
        ### probing the language model and obtaining the ranked tokens in the masked_position
        text, tokens, logprob = gpt3(prompt)  # TODO Figure out what the hell to do with probabilities
        probe_outputs = clean_up(text)

        # TODO: Check Logic consistency (Emile, Sel)

        ### saving the outputs and the likelihood scores received with the sample prompt
        results.append(
            {
                "Prompt": prompt,
                "SubjectEntity": subject_entity,
                "Relation": relation,
                "ObjectEntity": probe_outputs,
                "Probability": logprob,
            }
        )

        if index == SAMPLE_SIZE:
            break

    ### saving the prompt outputs separately for each relation type
    results_df = pd.DataFrame(results) #.sort_values(by=["SubjectEntity"], ascending=(True, False))

    if output_dir.exists():
        assert output_dir.is_dir()
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    results_df.to_csv(output_dir / f"{relation}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and Run the Baseline Method on Prompt Outputs"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./dev/",
        help="input directory containing the subject-entities for each relation to probe the language model",
    )
    parser.add_argument(
        "--baseline_output_dir",
        type=str,
        default="./gpt3_output/",
        help="output directory to store the baseline output",
    )
    args = parser.parse_args()
    print(args)

    input_dir = Path(args.input_dir)
    baseline_output_dir = Path(args.baseline_output_dir)

    ### call the prompt function to get output for each (subject-entity, relation)
    for relation in RELATIONS:
        entities = (
            pd.read_csv(input_dir / f"{relation}.csv")["SubjectEntity"]
                .drop_duplicates(keep="first")
                .tolist()
        )
        probe_lm(relation, entities, baseline_output_dir)


if __name__ == "__main__":
    main()
