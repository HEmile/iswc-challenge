import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

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


def initialize_lm(model_type, top_k):
    ### using the HuggingFace pipeline to initialize the model and its corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type).to(device)
    device_id = (
        -1 if device == torch.device("cpu") else 0
    )  ### -1 device is for cpu, 0 for gpu
    nlp = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=top_k, device=device_id
    )  ### top_k defines the number of ranked output tokens to pick in the [MASK] position
    return nlp, tokenizer.mask_token


def create_prompt(subject_entity, relation, mask_token):
    ### depending on the relation, we fix the prompt
    if relation == "CountryBordersWithCountry":
        prompt = subject_entity + " shares border with {}.".format(mask_token)
    elif relation == "CountryOfficialLanguage":
        prompt = (
                "The official language of "
                + subject_entity
                + " is {}.".format(mask_token)
        )
    elif relation == "StateSharesBorderState":
        prompt = subject_entity + " shares border with {} state.".format(mask_token)
    elif relation == "RiverBasinsCountry":
        prompt = subject_entity + " river basins in {}.".format(mask_token)
    elif relation == "ChemicalCompoundElement":
        prompt = subject_entity + " consits of {}, which is an element.".format(
            mask_token
        )
    elif relation == "PersonLanguage":
        prompt = subject_entity + " speaks in {}.".format(mask_token)
    elif relation == "PersonProfession":
        prompt = subject_entity + " is a {} by profession.".format(mask_token)
    elif relation == "PersonInstrument":
        prompt = subject_entity + " plays {}, which is an instrument.".format(
            mask_token
        )
    elif relation == "PersonEmployer":
        prompt = subject_entity + " is an employer at {}, which is a company.".format(
            mask_token
        )
    elif relation == "PersonPlaceOfDeath":
        prompt = subject_entity + " died at {}.".format(mask_token)
    elif relation == "PersonCauseOfDeath":
        prompt = subject_entity + " died due to {}.".format(mask_token)
    elif relation == "CompanyParentOrganization":
        prompt = (
                "The parent organization of "
                + subject_entity
                + " is {}.".format(mask_token)
        )
    return prompt


def probe_lm(model_type, top_k, relation, subject_entities, output_dir: Path):
    ### initializing the language model
    nlp, mask_token = initialize_lm(model_type, top_k)

    ### for every subject-entity in the entities list, we probe the LM using the below sample prompts
    results = []
    for subject_entity in subject_entities:
        print(
            "Probing the {} language model for {} (subject-entity) and {} relation".format(
                model_type, subject_entity, relation
            )
        )
        prompt = create_prompt(subject_entity, relation,
                               mask_token)  ### creating a specific prompt for the given relation
        probe_outputs = nlp(
            prompt)  ### probing the language model and obtaining the ranked tokens in the masked_position

        ### saving the top_k outputs and the likelihood scores received with the sample prompt
        for sequence in probe_outputs:
            results.append(
                {
                    "Prompt": prompt,
                    "SubjectEntity": subject_entity,
                    "Relation": relation,
                    "ObjectEntity": sequence["token_str"],
                    "Probability": round(sequence["score"], 4),
                }
            )

    ### saving the prompt outputs separately for each relation type
    results_df = pd.DataFrame(results).sort_values(
        by=["SubjectEntity", "Probability"], ascending=(True, False)
    )

    if output_dir.exists():
        assert output_dir.is_dir()
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    results_df.to_csv(output_dir / f"{relation}.csv", index=False)


def baseline(input_dir, prob_threshold, relations, output_dir: Path):
    print("Running the baseline method ...")

    ### for each relation, we run the baseline method
    for relation in relations:
        df = pd.read_csv(input_dir / f"{relation}.csv")
        df = df[
            df["Probability"] >= prob_threshold
            ]  ### all the output tokens with >= 0.5 likelihood are chosen and the rest are discarded

        if output_dir.exists():
            assert output_dir.is_dir()
        else:
            output_dir.mkdir(exist_ok=True, parents=True)

        df.to_csv(
            output_dir / f"{relation}.csv", index=False
        )  ### save the selected output tokens separately for each relation


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and Run the Baseline Method on Prompt Outputs"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-large-cased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./dev/",
        help="input directory containing the subject-entities for each relation to probe the language model",
    )
    parser.add_argument(
        "--prompt_output_dir",
        type=str,
        default="./prompt_output_bert_large_cased/",
        help="output directory to store the prompt output",
    )
    parser.add_argument(
        "--baseline_output_dir",
        type=str,
        default="./bert_large_output/",
        help="output directory to store the baseline output",
    )
    args = parser.parse_args()
    print(args)

    model_type = args.model_type
    input_dir = Path(args.input_dir)
    prompt_output_dir = Path(args.prompt_output_dir)
    baseline_output_dir = Path(args.baseline_output_dir)

    top_k = 100  ### picking the top 100 ranked prompt outputs in the [MASK] position

    ### call the prompt function to get output for each (subject-entity, relation)
    for relation in RELATIONS:
        entities = (
            pd.read_csv(input_dir / f"{relation}.csv")["SubjectEntity"]
                .drop_duplicates(keep="first")
                .tolist()
        )
        probe_lm(model_type, top_k, relation, entities, prompt_output_dir)

    prob_threshold = 0.5  ### setting the baseline threshold to select the output tokens

    ### run the baseline method on the prompt outputs
    baseline(prompt_output_dir, prob_threshold, RELATIONS, baseline_output_dir)


if __name__ == "__main__":
    main()
