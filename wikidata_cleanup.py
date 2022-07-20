# Go trough the output, check wheter what is produced is an alias for something and if it is, replace it by the prefered label
import argparse
import bz2
from collections import OrderedDict, defaultdict
import json
import pathlib
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple, Union, cast

from utils.file_io import read_lm_kbc_jsonl


class Database:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Tuple[Optional[str], int]]
                        ] = defaultdict(lambda: defaultdict(lambda: (None, -1)))
        # data maps from relation types to a dictionary from each alias to the proper labels
        # Note that there might be a collision where one alias is used for multiple things, hence the list
        self.labels: Dict[str, Dict[str, str]] = defaultdict(lambda: {})
        # defaultdict maps from relation types to known labels, where labels map from their lowercase version to cased

    def add_entry(self, relation: str, label: str, aliases: List[str], frequency: int):
        """Add an entry to this database"""
        relation_relevant = self.data[relation]
        for alias in aliases:
            alias_lower = alias.lower()
            alias_target = relation_relevant[alias_lower]
            if alias_target[1] < frequency:
                relation_relevant[alias_lower] = (label, frequency)
        self.labels[relation][label.lower()] = label

    def optional_main_label(self, relation: str, possible_alias: str) -> Optional[str]:
        return self.labels[relation].get(possible_alias.lower())

    def lookup(self, relation: str, possible_alias: str) -> Optional[str]:
        if relation not in self.data:
            return None
        possible_replacement = self.data[relation].get(possible_alias.lower())
        if not possible_replacement:
            return None
        return possible_replacement[0]


class WikiDataCleaner:
    def __init__(self, alias_file: pathlib.Path) -> None:
        self.database = Database()
        with bz2.open(alias_file, mode='rt') as f:
            for line in f:
                # {"r":["CountryBordersWithCountry","RiverBasinsCountry"],"l":"Belgium","a":["Kingdom of Belgium","BEL","be","🇧🇪","BE"],"c":240}
                entity_info = json.loads(line)
                label = entity_info["l"]
                aliases = entity_info["a"]
                claim_count = entity_info["c"]
                for relation in entity_info["r"]:
                    self.database.add_entry(relation, label, aliases, claim_count)

    def clean(self, original_prediction: Dict[str, Union[str, List[List[str]]]]) -> Dict[str, Union[str, List[str]]]:
        # {"SubjectEntity": "Acetone", "Relation": "ChemicalCompoundElement", "ObjectEntities": [["hydrogen"], ["carbon"], ["oxygen"]]}
        Relation: str = cast(str, original_prediction["Relation"])
        SubjectEntity: str = cast(str, original_prediction["SubjectEntity"])
        Prompt: str = cast(str, original_prediction["Prompt"])
        ObjectEntities: List[str] = cast(List[str], original_prediction["ObjectEntities"])
        corrected_ObjectEntities: Dict[str, str] = OrderedDict()  # Using to preserve the insertion order. This maps from lower case to a potentially case preserved variant
        for original_ObjectEntity in ObjectEntities:
            as_main = self.database.optional_main_label(Relation, original_ObjectEntity)
            if as_main:
                # The predicted one is a main label on wikidata, so we add it but keep the casings
                # corrected_ObjectEntities[original_ObjectEntity.lower()] = as_main
                corrected_ObjectEntities[original_ObjectEntity.lower()] = original_ObjectEntity
            else:
                possible_replacement = self.database.lookup(Relation, original_ObjectEntity)
                if possible_replacement:
                    # we put this replacement, note that this may merge multiple replacements resulting in the same label
                    corrected_ObjectEntities[possible_replacement.lower()] = possible_replacement
                else:
                    # not found, so we keep the original
                    corrected_ObjectEntities[original_ObjectEntity.lower()] = original_ObjectEntity
        return {"SubjectEntity": SubjectEntity, "Relation": Relation, "Prompt": Prompt, "ObjectEntities": list(corrected_ObjectEntities.values())}


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
        "-i",
        "--input-file",
        type=str,
        default="./predictions/gpt3.pred.jsonl",
        help="input directory containing the baseline or your method output",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="./predictions/gpt3_wikiclean.pred.jsonl",
        help="Output file (required)",
    )
    parser.add_argument(
        "-c",
        "--cleaning-rules",
        type=str,
        default="./aliases.jsonl.bz2",
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
