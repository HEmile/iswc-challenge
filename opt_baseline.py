import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.file_io import read_lm_kbc_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 5000
BATCH_SIZE = 1


def clean_up(probe_outputs, prompt):
    """ functions to clean up api output """
    probe_outputs = probe_outputs.replace(prompt, '')
    probe_outputs = probe_outputs.replace('%', '').strip()
    if probe_outputs == 'None':
        return []
    else:
        probe_outputs = probe_outputs.split("', '")
    return probe_outputs


def create_prompt(subject_entity, relation):
    ### depending on the relation, we fix the prompt
    if relation == "CountryBordersWithCountry":
        prompt = f"""
Which countries neighbour Dominica?
Venezuela%
Which countries neighbour North Korea?
South Korea, China, Russia%
Which countries neighbour Serbia?
Montenegro, Kosovo, Bosnia and Herzegovina, Hungary, Croatia, Bulgaria,  Macedonia, Albania, Romania%
Which countries neighbour Fiji?
None%
Which countries neighbour {subject_entity}?
"""

    elif relation == "CountryOfficialLanguage":
        prompt = f"""
Which are the official languages of Suriname?
Dutch%
Which are the official languages of Canada?
English, French%
Which are the official languages of Singapore?
English, Malay, Mandarin, Tamil%
Which are the official languages of Sri Lanka?
Sinhala, Tamil%
Which are the official languages of {subject_entity}?        
"""
    elif relation == "StateSharesBorderState":
        prompt = f"""
What states border San Marino?
San Leo, Acquaviva, Borgo Maggiore, Chiesanuova, Fiorentino%
What states border Whales?
England%
What states border Liguria?
Tuscany, Auvergne-Rhoone-Alpes, Piedmont, Emilia-Romagna%
What states border Mecklenberg-Western Pomerania?
Brandenburg, Pomeranian, Schleswig-Holstein, Lower Saxony%
What states border {subject_entity}?
"""

    elif relation == "RiverBasinsCountry":
        prompt = f"""
What countries does the river Drava cross?
Hungary, Italy, Austria, Slovenia, Croatia%
What countries does the river Huai river cross?
China%
What countries does the river Paraná river cross?
Bolivia, Paraguay, Argentina, Brazil%
What countries does the river Oise cross?
Belgium, France%
What countries does the river {subject_entity} cross?
"""

    elif relation == "ChemicalCompoundElement":
        prompt = f"""
What are all the chemical elements that make up the molecule Water?
Hydrogen, Oxygen%
What are all the chemical elements that make up the molecule Bismuth subsalicylate?
Bismuth%
What are all the chemical elements that make up the molecule Sodium Bicarbonate?
Hydrogen, Oxygen, Sodium, Carbon%
What are all the chemical elements that make up the molecule Aspirin?
Oxygen, Carbon, Hydrogen%
What are all the chemical elements that make up the molecule {subject_entity}?
"""
    elif relation == "PersonLanguage":
        prompt = f"""
Which languages does Aamir Khan speak?
Hindi, English, Urdu%
Which languages does Pharrell Williams speak?
English%
Which languages does Xabi Alonso speak?
Catalan, English, Portuguese, Spanish%
Which languages does Shakira speak?
Catalan, English, Portuguese, Spanish, Italian, French%
Which languages does {subject_entity} speak?
"""

    elif relation == "PersonProfession":
        prompt = f"""
What is Danny DeVito's profession?
Comedian, Film Director, Voice Actor, Actor, Film Producer, Film Actor, Dub Actor, Activist, Television Actor%
What is David Guetta's profession?
DJ%
What is Gary Lineker's profession?
Commentator, Association Football Player, Journalist, Broadcaster%
What is Gwyneth Paltrow's profession?
Film Actor, Musician%
What is {subject_entity}'s profession?
"""

    elif relation == "PersonInstrument":
        prompt = f"""
Which instruments does Liam Gallagher play?
Maraca, Guitar%
Which instruments does Jay Park play?
None%
Which instruments does Axl Rose play?
Guitar, Piano, Pander, Bass%
Which instruments does Neil Young play?
Guitar%
Which instruments does {subject_entity} play?
"""
    elif relation == "PersonEmployer":
        prompt = f"""
Where is or was Susan Wojcicki employed?
Google%
Where is or was Steve Wozniak employed?
Apple Inc, Hewlett-Packard, University of Technology Sydney, Atari%
Where is or was Yukio Hatoyama employed?
Senshu University,Tokyo Institute of Technology%
Where is or was Yahtzee Croshaw employed?
PC Gamer, Hyper, Escapist%
Where is or was {subject_entity} employed?
"""
    elif relation == "PersonPlaceOfDeath":
        prompt = f"""
What is the place of death of Barack Obama?
None%
What is the place of death of Ennio Morricone?
Rome%
What is the place of death of Elon Musk?
None%
What is the place of death of Prince?
Chanhassen%
What is the place of death of {subject_entity}? 
"""

    elif relation == "PersonCauseOfDeath":
        prompt = f"""
How did André Leon Talley die?
Infarction%
How did Angela Merkel die?
None%
How did Bob Saget die?
Injury, Blunt Trauma%
How did Jamal Khashoggi die?
Murder%
How did {subject_entity} die? 
"""

    elif relation == "CompanyParentOrganization":
        prompt = f"""
What is the parent company of Microsoft?
None%
What is the parent company of Sony?
Sony Group%
What is the parent company of Saab?
Saab Group, Saab-Scania, Spyker N.V., National Electric Vehicle Sweden, General Motors%
What is the parent company of Max Motors?
None%
What is the parent company of {subject_entity}?
"""
    return prompt


def predict(model, tokenizer, prompts):
    if torch.cuda.is_available():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(0)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    # get length of input
    # input_length = len(inputs["input_ids"].tolist()[0])
    # print("Input lengths {}".format(len(inputs)))
    output = model.generate(**inputs,
                            max_length=512, eos_token_id=int(tokenizer.convert_tokens_to_ids("%")))
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

    return generated_texts


def probe_lm(input: Path, model_name, output: Path, batch_size=BATCH_SIZE):
    ### for every subject-entity in the entities list, we probe the LM using the below sample prompts

    # Load the input file
    logger.info(f"Loading the input file \"{input}\"...")
    input_rows = read_lm_kbc_jsonl(input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Trim list & batch entities
    input_rows = input_rows[:SAMPLE_SIZE]  #
    batches = [input_rows[x:x + batch_size] for x in range(0, len(input_rows), batch_size)]

    # Load HF model and tokenizer
    logger.info(f"Loading {model_name} from HF model hub.")
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # eos_token_id = int(tokenizer.convert_tokens_to_ids("%")),
    # tokenizer.pad_token = tokenizer.eos_token

    results = []
    for idx, batch in tqdm(enumerate(batches)):
        prompts = []
        for index, row in enumerate(batch):
            # TODO: Generate examples in the prompt automatically (Thiviyan)
            #
            # TODO: Rephrase prompt automatically (Dimitris)

            ### creating a specific prompt for the given relation
            logger.info(f"Creating prompts...")
            prompts.append(create_prompt(row['SubjectEntity'], row['Relation']))

        ### probing the language model and obtaining the ranked tokens in the masked_position
        # logger.info(f"Running the model...")

        predictions = predict(model, tokenizer, prompts)  # TODO Figure out what to do with probabilities

        ### Clean and format results
        for row, prediction, prompt in zip(batch, predictions, prompts):
            prediction = clean_up(prediction, prompt)
            # TODO: Check Logic consistency (Emile, Sel)

            result = {
                "SubjectEntity": row['SubjectEntity'],
                "Relation": row['Relation'],
                "Prompt": prompt,
                "ObjectEntities": prediction
            }
            results.append(result)

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
        "--i",
        "--input",
        type=str,
        default="data/dev.jsonl",
        help="input file containing the subject-entities for each relation to probe the language model",
    )
    parser.add_argument(
        "--m",
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="input the model name from the HF model hub",
    )
    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default="predictions/opt-1.3b.pred.jsonl",
        help="output directory to store the baseline output",
    )
    args = parser.parse_args()
    print(args)

    probe_lm(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
