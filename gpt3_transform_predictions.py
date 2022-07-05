import argparse
import json
import logging
from pathlib import Path

import pandas as pd

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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def restructure(input_dir: Path, output_dir: Path):
    ### looping over all the files in the input directory
    all_dfs = []
    for fname in input_dir.glob("*.csv"):
        prompt_df = pd.read_csv(fname)

        all_dfs.append(prompt_df)

    full_df = pd.concat(all_dfs)
    full_df = full_df.groupby(["SubjectEntity",
                               "Relation"]).agg(ObjectEntities=('ObjectEntity', 'unique')).reset_index()
    full_df['ObjectEntities'] = full_df['ObjectEntities'].apply(lambda x: list(x))

    # Save the results
    logger.info(f"Saving the results to \"{output_dir}\"...")
    with open(output_dir, "w") as f:
        # for result in results:
        for index, result in full_df.iterrows():
            result = result.to_dict()
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./predictions/gpt3_output/",
        help="input directory containing the baseline or your method output",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./predictions/gpt3.pred.jsonl",
        help="Output file (required)",
    )

    args = parser.parse_args()
    print(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    assert input_dir.exists() and input_dir.is_dir()

    restructure(input_dir, output_dir)


if __name__ == "__main__":
    main()
