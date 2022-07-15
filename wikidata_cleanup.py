# Go trough the output, check wheter what is produced is an alias for something and if it is, replace it by the prefered label
import argparse
from cProfile import label
from collections import defaultdict
from distutils.command.clean import clean
from importlib.resources import path
import json
import pathlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils.file_io import read_lm_kbc_jsonl


class Database:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, list[str]]
                        ] = defaultdict(lambda: defaultdict(list))
        # data maps from relation types to a dictionary from each alias to the proper labels
        # Note that there might be a collision where one alias is used for multiple things, hence the list
        self.labels: dict[str, set[str]] = defaultdict(set)
        # defaultdict maps from relation types to known labels

    def add_entry(self, relation: str, label: str, aliases: List[str]):
        """Add an entry to this database"""
        for alias in aliases:
            self.data[relation][alias.lower()].append(label.lower())
        self.labels[relation].add(label.lower())

    def is_main_label(self, relation: str, possible_alias: str) -> Optional[str]:
        return possible_alias.lower() in self.labels[relation]

    def lookup(self, relation: str, possible_alias: str) -> Optional[List[str]]:
        if relation not in self.data:
            return None
        if possible_alias.lower() not in self.data[relation]:
            return None
        return self.data[relation][possible_alias.lower()]


class WikiDataCleaner:
    def __init__(self, alias_file: pathlib.Path) -> None:
        self.database = Database()
        with open(alias_file) as f:
            for line in f:
                # {"r":["CountryBordersWithCountry","RiverBasinsCountry"],"l":"Belgium","a":["Kingdom of Belgium","BEL","be","🇧🇪","BE"]}
                entity_info = json.loads(line)
                label = entity_info["l"]
                aliases = entity_info["a"]
                for relation in entity_info["r"]:
                    self.database.add_entry(relation, label, aliases)

    def clean(self, original_prediction: Dict[str, Union[str, List[List[str]]]]) -> Dict[str, Union[str, List[List[str]]]]:
        # {"SubjectEntity": "Acetone", "Relation": "ChemicalCompoundElement", "ObjectEntities": [["hydrogen"], ["carbon"], ["oxygen"]]}
        Relation = original_prediction["Relation"]
        SubjectEntity = original_prediction["SubjectEntity"]
        Prompt = original_prediction["Prompt"]
        ObjectEntities = original_prediction["ObjectEntities"]
        corrected_ObjectEntities = []
        # TODO optionally we could choose to act differently dependent on the size of the list
        for original_ObjectEntity in ObjectEntities:
            if self.database.is_main_label(Relation, original_ObjectEntity):
                corrected_ObjectEntities.append(original_ObjectEntity)
            else:
                possible_replacement = self.database.lookup(Relation, original_ObjectEntity)
                if possible_replacement:
                    for replacements in possible_replacement:
                        corrected_ObjectEntities.append(replacements)
                else:
                    corrected_ObjectEntities.append(original_ObjectEntity)
        return {"SubjectEntity": SubjectEntity, "Relation": Relation, "Prompt": Prompt, "ObjectEntities": corrected_ObjectEntities}


def wikidata_clean(input_file: pathlib.Path, output_file: pathlib.Path, cleaning_rules: pathlib.Path):
    # looping over all the files in the input directory
    original_predictions = read_lm_kbc_jsonl(input_file)
    cleaner = WikiDataCleaner(cleaning_rules)
    with open(output_file, "w") as f:
        for original_prediction in original_predictions:
            cleaned_prediction = cleaner.clean(original_prediction)
            f.write(json.dumps(cleaned_prediction) + "\n")


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
        default="./predictions/gpt3_wikiclean.pred.jsonl",
        help="Output file (required)",
    )
    parser.add_argument(
        "-c",
        "--cleaning-rules",
        type=str,
        default="./aliases_classes.txt",
        help="File with aliases (required)",
    )

    args = parser.parse_args()
    print(args)

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    cleaning_rules = Path(args.cleaning_rules)

    assert input_file.exists()
    assert cleaning_rules.exists()

    wikidata_clean(input_file, output_file, cleaning_rules)


if __name__ == "__main__":
    main()
