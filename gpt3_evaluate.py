import argparse
import pprint
import sys
from ast import literal_eval
from pathlib import Path

import numpy as np
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


### Precision = how much of the LM's predictions match with ground truth (GT) - |LM intersection GT| / |LM|
def precision(x, y):
    count = 0
    for pred in x:
        ### if the prediction is substring of any of the ground truth object-entity strings, count is incremented
        count += 1 if any(pred in string for string in y) else 0
    return count / len(x)


### Recall = how much of the LM's predictions match are within ground truth (GT) - |LM intersection GT| / |GT|
def recall(x, y):
    count = 0
    for pred in x:
        ### if the prediction is substring of any of the ground truth object-entity strings, count is incremented
        count += 1 if any(pred in string for string in y) else 0
    return count / len(y)


#### ref: https://en.wikipedia.org/wiki/F-score
#### F1 = (2 * P * R)) / (P + R)
def f1_score(x, y):
    p = precision(x, y)
    r = recall(x, y)
    if p == r == 0:
        return 0, 0, 0
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


### converting the LM predictions into lower case and removing punctuations
def clean_predictions(x):
    if type(x) == float:
        return
    elif x == 'NONE':
        return x
    else:
        return x.lower().strip().replace(".", "").replace(",", "").replace("-", "")


def evaluate(input_dir: Path, ground_truth_dir: Path, results_dir: Path):
    if results_dir.exists():
        assert results_dir.is_dir()
    else:
        results_dir.mkdir(parents=True, exist_ok=True)

    # dictionary (key:relation, val:F1-score)
    # to store average F1 scores across all subject-entities for a given relation
    average_metrics = {'f1': {}, 'recall': {}, 'precision': {}}

    ### looping over all the files in the input directory
    for fname in input_dir.glob("*.csv"):
        prompt_df = pd.read_csv(fname)
        ### getting the relation name from the file name
        relation = fname.stem

        if relation not in RELATIONS:
            sys.stderr.write(f'\nWARNING: Ignored" '
                             f'Filename: "{fname.absolute()}" --- "{relation}" is not a valid relation\n\n')
            continue

        ground_truth_df = pd.read_csv(ground_truth_dir / f"{relation}.csv")
        ground_truth_df["ObjectEntity"] = ground_truth_df["ObjectEntity"].apply(literal_eval)

        # looping through entities
        res_df = []
        for entity, ground_truth_objects in ground_truth_df.groupby(["SubjectEntity"])["ObjectEntity"]:
            ground_truth_objects = ground_truth_objects.tolist()
            predictions = prompt_df[prompt_df["SubjectEntity"] == entity]["ObjectEntity"].tolist()
            print('SubjectEntity: %s' % entity, 'Ground Truth: %s' % ground_truth_objects,
                  'Predictions: %s' % predictions)

            if len(predictions) == 0:
                ### if no predictions obtained for the subject-entity, then F1-score is 0
                # TODO bug?
                res_df.append({"SubjectEntity": entity, "Relation": relation,
                               "Precision": 0, "Recall": 0, "F1-score": 0})
            else:
                predictions = [clean_predictions(x) for x in predictions]
                # calculating F1-score for the given subject-entity
                prec, reca, f1 = f1_score(predictions, ground_truth_objects)
                res_df.append({"SubjectEntity": entity, "Relation": relation,
                               "Precision": prec, "Recall": reca, "F1-score": f1})

        ### storing the results separately for each relation
        res_df = pd.DataFrame(res_df)
        res_df.to_csv(results_dir / f"{relation}_results.csv", index=False)

        ### calculating the averaged F1-score across all subject-entities
        average_metrics['f1'][relation] = res_df["F1-score"].mean()
        average_metrics['recall'][relation] = res_df["Recall"].mean()
        average_metrics['precision'][relation] = res_df["Precision"].mean()

    ### calculating the final F1-score averaged across all the relations
    ### NOTE: this score will be used to rank the participating systems
    f1_total = round(np.mean(list(average_metrics['f1'].values())) * 100, 2)
    pr_total = round(np.mean(list(average_metrics['precision'].values())) * 100, 2)
    re_total = round(np.mean(list(average_metrics['recall'].values())) * 100, 2)

    print("average F1-score for each relation:")
    pprint.pprint(average_metrics)

    print(f"Final F1-score: {f1_total} %")
    print(f"Final Precision-score: {pr_total} %")
    print(f"Final Recall-score: {re_total} %")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./bert_large_output/",
        help="input directory containing the baseline or your method output",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="./dev/",
        help="ground truth directory containing true object-entities for the subject-entities for which the LM was probed and then baseline or your method was applied",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./bert_large_results/",
        help="results directory for storing the F1 scores for baseline or your method",
    )
    args = parser.parse_args()
    print(args)

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.ground_truth_dir)

    assert input_dir.exists() and input_dir.is_dir()
    assert gt_dir.exists() and gt_dir.is_dir()

    results_dir = Path(args.results_dir)

    evaluate(input_dir, gt_dir, results_dir)


if __name__ == "__main__":
    main()
