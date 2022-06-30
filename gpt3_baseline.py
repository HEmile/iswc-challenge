import argparse
import time
from pathlib import Path

import pandas as pd

from utils.model import gpt3

SAMPLE_SIZE = 200

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


def clean_up(probe_outputs):
    """ functions to clean up api output """
    probe_outputs = probe_outputs.strip()
    probe_outputs = probe_outputs[2:-2].split("', '")
    return probe_outputs


def convert_nan(probe_outputs):
    new_probe_outputs = []
    for item in probe_outputs:
        if item == 'NONE':
            new_probe_outputs.append(None)
        else:
            new_probe_outputs.append(item)
    return new_probe_outputs


def create_prompt(subject_entity, relation):
    ### depending on the relation, we fix the prompt
    if relation == "CountryBordersWithCountry":
        prompt = f"""
Which countries neighbour Dominica?
['Venezuela']

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
Which are the official languages of Suriname?
['Dutch']

Which are the official languages of Canada?
['English', 'French']

Which are the official languages of Singapore?
['English', 'Malay', 'Mandarin', 'Tamil']

Which are the official languages of Sri Lanka?
['Sinhala', 'Tamil']

Which are the official languages of {subject_entity}?        
"""
    elif relation == "StateSharesBorderState":
        prompt = f"""
What states border San Marino?
['San Leo', 'Acquaviva', 'Borgo Maggiore', 'Chiesanuova', 'Fiorentino']

What states border Whales?
['England']

What states border Liguria?
['Tuscany', 'Auvergne-Rhoone-Alpes', 'Piedmont', 'Emilia-Romagna']

What states border Mecklenberg-Western Pomerania?
['Brandenburg', 'Pomeranian', 'Schleswig-Holstein', 'Lower Saxony']

What states border {subject_entity}?
"""

    elif relation == "RiverBasinsCountry":
        prompt = f"""
What countries does the river Drava cross?
['Hungary', 'Italy', 'Austria', 'Slovenia', 'Croatia']

What countries does the river Huai river cross?
['China']

What countries does the river Paraná river cross?
['Bolivia', 'Paraguay', 'Argentina', 'Brazil']

What countries does the river Oise cross?
['Belgium', 'France']

What countries does the river {subject_entity} cross?
"""

    elif relation == "ChemicalCompoundElement":
        prompt = f"""
What are all the chemical elements that make up the molecule Water?
['Hydrogen', 'Oxygen']

What are all the chemical elements that make up the molecule Bismuth subsalicylate	?
['Bismuth']

What are all the chemical elements that make up the molecule Sodium Bicarbonate	?
['Hydrogen', 'Oxygen', 'Sodium', 'Carbon']

What are all the chemical elements that make up the molecule Aspirin?
['Oxygen', 'Carbon', 'Hydrogen']

What are all the chemical elements that make up the molecule {subject_entity}?
"""
    elif relation == "PersonLanguage":
        prompt = f"""
Which languages does Aamir Khan speak?
['Hindi', 'English', 'Urdu']

Which languages does Pharrell Williams speak?
['English']

Which languages does Shakira speak?
['Catalan', 'English', 'Portuguese', 'Spanish']

Which languages does Shakira speak?
['Catalan', 'English', 'Portuguese', 'Spanish', 'Italian', 'French']

Which languages does {subject_entity} speak?
"""

    elif relation == "PersonProfession":
        prompt = f"""
What is Danny DeVito's profession?
['Comedian', 'Film Director', 'Voice Actor', 'Actor', 'Film Producer', 'Film Actor', 'Dub Actor', 'Activist', 'Television Actor' ] 

What is David Guetta's profession?
['DJ']

What is Gary Lineker's profession?
['Commentator', 'Association Football Player', 'Journalist', 'Broadcaster']

What is Gwyneth Paltrow's profession?
['Film Actor','Musician']

What is {subject_entity}'s profession?
"""

    elif relation == "PersonInstrument":
        prompt = f"""
Which instruments does Liam Gallagher play?
['Maraca', 'Guitar']

Which instruments does Jay Park play?
['NONE']

Which instruments does Axl Rose play?
['Guitar', 'Piano', 'Pander', 'Bass']

Which instruments does Neil Young play?
['Guitar']

Which instruments does {subject_entity} play?
"""
    elif relation == "PersonEmployer":
        prompt = f"""
Where is or was Susan Wojcicki employed?
['Google']

Where is or was Steve Wozniak employed?
['Apple Inc', 'Hewlett-Packard', 'University of Technology Sydney', 'Atari']

Where is or was Yukio Hatoyama employed?
['Senshu University','Tokyo Institute of Technology']

Where is or was Yahtzee Croshaw employed?
['PC Gamer', 'Hyper', 'Escapist']

Where is or was {subject_entity} employed?
"""
    elif relation == "PersonPlaceOfDeath":
        prompt = f"""
What is the place of death of Barack Obama?
['NONE']

What is the place of death of Ennio Morricone?
['Rome']

What is the place of death of Elon Musk?
['NONE']

What is the place of death of Prince?
['Chanhassen']

What is the place of death of {subject_entity}? 
"""

    elif relation == "PersonCauseOfDeath":
        prompt = f"""
How did André Leon Talley die?
['Infarction']

How did Angela Merkel die?
['NONE']

How did Bob Saget die?
['Injury', 'Blunt Trauma']

How did Jamal Khashoggi die?
['Murder']

How did {subject_entity} die? 
"""

    elif relation == "CompanyParentOrganization":
        prompt = f"""
What is the parent company of Microsoft?
['NONE']

What is the parent company of Sony?
['Sony Group']

What is the parent company of Saab?
['Saab Group', 'Saab-Scania', 'Spyker N.V.', 'National Electric Vehicle Sweden'', 'General Motors']

What is the parent company of Max Motors?
['NONE']

What is the parent company of {subject_entity}?
"""
    return prompt


def probe_lm(relation, subject_entities, output_dir: Path, batch_size=20):
    ### for every subject-entity in the entities list, we probe the LM using the below sample prompts

    # Trim list & batch entities
    subject_entities = subject_entities[:SAMPLE_SIZE]
    batches = [subject_entities[x:x + batch_size] for x in range(0, len(subject_entities), batch_size)]

    results = []
    for idx, batch in enumerate(batches):
        prompts = []
        for index, subject_entity in enumerate(batch):
            print(f"Probing the GPT3 language model "
                  f"for {subject_entity} (subject-entity) and {relation} relation")

            # TODO: Generate examples in the prompt automatically (Thiviyan)
            #
            # TODO: Rephrase prompt automatically (Dimitris)

            ### creating a specific prompt for the given relation
            prompts.append(create_prompt(subject_entity, relation))

        ### probing the language model and obtaining the ranked tokens in the masked_position
        predictions = gpt3(prompts)  # TODO Figure out what the hell to do with probabilities

        for prediction in predictions:
            prediction['text'] = clean_up(prediction['text'])
            prediction['text'] = convert_nan(prediction['text'])

        # TODO: Check Logic consistency (Emile, Sel)

        ### saving the outputs and the likelihood scores received with the sample prompt
        x = [
            {
                "Prompt": prompts[index],
                "SubjectEntity": subject_entity,
                "Relation": relation,
                "ObjectEntity": predictions[index]['text'],
                "Probability": predictions[index]['logprob'],
            }
            for index, subject_entity in enumerate(batch)
        ]
        results += x

        # Sleep is needed becase we make many API calls. We can make 60 calls every minute
        if idx % 5:
            time.sleep(5)

    ### saving the prompt outputs separately for each relation type
    results_df = pd.DataFrame(results)  # .sort_values(by=["SubjectEntity"], ascending=(True, False))

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
