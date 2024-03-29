import argparse
import json
import logging
import time
from pathlib import Path

from tqdm.auto import tqdm

from utils.file_io import read_lm_kbc_jsonl
from utils.model import gpt3, clean_up, convert_nan

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 5000

MODEL_TYPES = ['text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001']


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
[]

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
What are all the atoms that make up the molecule Water?
['Hydrogen', 'Oxygen']

What are all the atoms that make up the molecule Bismuth subsalicylate	?
['Bismuth']

What are all the atoms that make up the molecule Sodium Bicarbonate	?
['Hydrogen', 'Oxygen', 'Sodium', 'Carbon']

What are all the atoms that make up the molecule Aspirin?
['Oxygen', 'Carbon', 'Hydrogen']

What are all the atoms that make up the molecule {subject_entity}?
"""
    elif relation == "PersonLanguage":
        prompt = f"""
Which languages does Aamir Khan speak?
['Hindi', 'English', 'Urdu']

Which languages does Pharrell Williams speak?
['English']

Which languages does Xabi Alonso speak?
['German', 'Basque', 'Spanish', 'English']

Which languages does Shakira speak?
['Catalan', 'English', 'Portuguese', 'Spanish', 'Italian', 'French']

Which languages does {subject_entity} speak?
"""

    elif relation == "PersonProfession":
        prompt = f"""
What is Danny DeVito's profession?
['Comedian', 'Film Director', 'Voice Actor', 'Actor', 'Film Producer', 'Film Actor', 'Dub Actor', 'Activist', 'Television Actor']
 
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
[]

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
[]

What is the place of death of Ennio Morricone?
['Rome']

What is the place of death of Elon Musk?
[]

What is the place of death of Prince?
['Chanhassen']

What is the place of death of {subject_entity}? 
"""

    elif relation == "PersonCauseOfDeath":
        prompt = f"""
How did André Leon Talley die?
['Infarction']

How did Angela Merkel die?
[]

How did Bob Saget die?
['Injury', 'Blunt Trauma']

How did Jamal Khashoggi die?
['Murder']

How did {subject_entity} die?
"""

    elif relation == "CompanyParentOrganization":
        prompt = f"""
What is the parent company of Microsoft?
[]

What is the parent company of Sony?
['Sony Group']

What is the parent company of Saab?
['Saab Group', 'Saab-Scania', 'Spyker N.V.', 'National Electric Vehicle Sweden', 'General Motors']

What is the parent company of Max Motors?
[]

What is the parent company of {subject_entity}?
"""
    return prompt


def load_prompt(subject_entity, relation, prompt_type='triple'):
    """
        Function that loads the prompts from the .txt files

    :param subject_entity:
    :param relation:
    :param simple: Set to True if you want to load the triple-based prompts (a.k.a. simple prompts)
    :return:
    """
    if prompt_type == 'triple':
        prompt_path = Path('data/prompts_triple_based')
    elif prompt_type == 'language':
        prompt_path = Path('data/prompts_natural_language')
    else:
        prompt_path = Path('data/prompts_optimized')

    prompt_path = Path.joinpath(prompt_path, f"{relation}.txt")
    with open(prompt_path, "r") as f:
        prompt = f.read()

    prompt = prompt.format(subject_entity=subject_entity)

    return prompt


def probe_lm(input: Path, model: str, output: Path, prompt_type='triple', batch_size=20):
    ### for every subject-entity in the entities list, we probe the LM using the below sample prompts

    # Load the input file
    logger.info(f"Loading the input file \"{input}\"...")
    input_rows = read_lm_kbc_jsonl(input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Trim list & batch entities
    input_rows = input_rows[:SAMPLE_SIZE]  #
    batches = [input_rows[x:x + batch_size] for x in range(0, len(input_rows), batch_size)]

    results = []
    for idx, batch in tqdm(enumerate(batches)):
        prompts = []
        for index, row in enumerate(batch):
            # TODO: Generate examples in the prompt automatically (Thiviyan)
            #
            # TODO: Rephrase prompt automatically (Dimitris)

            ### creating a specific prompt for the given relation
            logger.info(f"Creating prompts...")
            prompts.append(load_prompt(row['SubjectEntity'], row['Relation'], prompt_type))

        ### probing the language model and obtaining the ranked tokens in the masked_position
        logger.info(f"Running the model...")
        predictions = gpt3(prompts, model=model)  # TODO Figure out what to do with probabilities

        ### Clean and format results
        for row, prediction in zip(batch, predictions):
            prediction['text'] = clean_up(prediction['text'])
            prediction['text'] = convert_nan(prediction['text'])

            result = {
                "SubjectEntity": row['SubjectEntity'],
                "Relation": row['Relation'],
                "Prompt": prediction['prompt'],
                "ObjectEntities": prediction['text']
            }
            results.append(result)

        # Sleep is needed because we make many API calls. We can make 60 calls every minute
        if idx % 5:
            time.sleep(2)

    ### saving the prompt outputs separately for each relation type
    logger.info(f"Saving the results to \"{output}\"...")
    with open(output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and Run the Baseline Method on Prompt Outputs"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/dev.jsonl",
        help="input file containing the subject-entities for each relation to probe the language model",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions/gpt3.pred.jsonl",
        help="output directory to store the baseline output",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="text-davinci-002",
        help="The models provided by OpenAI. \
        Options: 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="triple",
        help="Simple vs natural language vs optimized based prompts\
        Options: 'triple', 'language', 'optimized'",
    )
    args = parser.parse_args()
    print(args)

    probe_lm(args.input, args.model, args.output, args.prompt_type)


if __name__ == "__main__":
    main()
