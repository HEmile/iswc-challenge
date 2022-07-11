import argparse
import json
import logging

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

from utils.file_io import read_lm_kbc_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PromptSet(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> T_co:
        return self.prompts[index]


def create_prompt(subject_entity: str, relation: str, mask_token: str) -> str:
    """
    Depending on the relation, we fix the prompt
    """

    prompt = mask_token

    if relation == "CountryBordersWithCountry":
        prompt = f"{subject_entity} shares border with {mask_token}."
    elif relation == "CountryOfficialLanguage":
        prompt = f"The official language of {subject_entity} is {mask_token}."
    elif relation == "StateSharesBorderState":
        prompt = f"{subject_entity} shares border with {mask_token} state."
    elif relation == "RiverBasinsCountry":
        prompt = f"{subject_entity} river basins in {mask_token}."
    elif relation == "ChemicalCompoundElement":
        prompt = f"{subject_entity} consists of {mask_token}, " \
                 f"which is an element."
    elif relation == "PersonLanguage":
        prompt = f"{subject_entity} speaks in {mask_token}."
    elif relation == "PersonProfession":
        prompt = f"{subject_entity} is a {mask_token} by profession."
    elif relation == "PersonInstrument":
        prompt = f"{subject_entity} plays {mask_token}, which is an instrument."
    elif relation == "PersonEmployer":
        prompt = f"{subject_entity} is an employer at {mask_token}, " \
                 f"which is a company."
    elif relation == "PersonPlaceOfDeath":
        prompt = f"{subject_entity} died at {mask_token}."
    elif relation == "PersonCauseOfDeath":
        prompt = f"{subject_entity} died due to {mask_token}."
    elif relation == "CompanyParentOrganization":
        prompt = f"The parent organization of {subject_entity} is {mask_token}."

    return prompt


def run(args):
    # Load the model
    model_type = args.model
    logger.info(f"Loading the model \"{model_type}\"...")

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type)

    pipe = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=args.gpu
    )

    mask_token = tokenizer.mask_token

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = PromptSet([create_prompt(
        subject_entity=row["SubjectEntity"],
        relation=row["Relation"],
        mask_token=mask_token,
    ) for row in input_rows])

    # Run the model
    logger.info(f"Running the model...")
    outputs = []
    for out in tqdm(pipe(prompts, batch_size=8), total=len(prompts)):
        outputs.append(out)
    results = []
    for row, prompt, output in zip(input_rows, prompts, outputs):
        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Prompt": prompt,
            "ObjectEntities": [
                seq["token_str"]
                for seq in output if seq["score"] > args.threshold],
        }
        results.append(result)

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and "
                    "Run the Baseline Method on Prompt Outputs"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-large-cased",
        help="HuggingFace model name (default: bert-large-cased)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/dev.jsonl",
        required=True,
        help="Input test file (required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions/baseline.pred.jsonl",
        required=True,
        help="Output file (required)",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Top k prompt outputs (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=-1,
        help="GPU ID, (default: -1, i.e., using CPU)"
    )

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
