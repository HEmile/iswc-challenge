import argparse
import string
from typing import List, Dict, Union
import sys
import pandas as pd

from utils.file_io import read_lm_kbc_jsonl
from pathlib import Path


def clean_object(obj: str) -> Union[str, None]:
    """
    Cleans the object by removing punctuation and lower-casing.
    """

    if not obj:
        return None

    for punctuation in string.punctuation:
        obj = obj.replace(punctuation, "")

    return obj.lower().strip()


def is_none_gts(gts: List[List[str]]) -> bool:
    """
    Checks if the ground truth object is none.
    """
    return not gts


def is_none_preds(preds: List[str]) -> bool:
    """
    Checks if the prediction object is none (with relaxing rules).
    """

    return preds is None or len(preds) == 0 or (len(preds) == 1 and (
            list(preds)[0] is None or
            list(preds)[0].lower() in {"", "none", "null"}))


def true_positives(preds: List[str], gts: List[List[str]]) -> int:
    """
    Calculates the number of true positives
    for a given pair of subject and relation.
    Method:
        Iterate over the ground truth objects, each is a list of possible
        aliases. For each ground truth object, check if the prediction
        contains any of its aliases. If so, increment the true positives by 1.

    Args:
        preds: list of normalized predictions
        gts: list of ground truth objects (lists of normalized aliases)

    Returns:
        true_positives: int
    """

    tp = 0
    for gt in gts:
        gt_set = set(gt)
        if any(pred in gt_set for pred in preds):
            tp += 1

    return tp


def precision(preds: List[str], gts: List[List[str]]) -> float:
    """
    Calculates the precision of the predictions
    for a given pair of subject and relation.

    Args:
        preds: list of predictions
        gts: list of ground truth objects

    Returns:
        precision: float
    """

    # When the ground truth object is none
    if is_none_gts(gts):
        return 1.0 if is_none_preds(preds) else 0.0

    # When the ground truth object is not none
    try:
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except (ZeroDivisionError, TypeError):
        return 0.0


def recall(preds: List[str], gts: List[List[str]]) -> float:
    """
    Calculates the recall of the predictions
    for a given pair of subject and relation.

    Args:
        preds: list of predictions
        gts: list of ground truth objects

    Returns:
        recall: float
    """

    # When the ground truth object is none
    if is_none_gts(gts):
        return 1.0 if is_none_preds(preds) else 0.0

    # When the ground truth object is not none
    try:
        return true_positives(preds, gts) / len(gts)
    except TypeError:
        return 0.0


def f1_score(p: float, r: float) -> float:
    """
    Calculates the F1-score of the predictions
    for a given pair of subject and relation.

    Args:
        p: precision
        r: recall

    Returns:
        f1_score: float
    """

    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    """
    Index the ground truth/prediction rows by subject entity and relation.
    """

    return {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in
            rows}


def evaluate_per_sr_pair(predictions_fp, ground_truth_fp) -> List[Dict[str, float]]:
    pred_rows = read_lm_kbc_jsonl(predictions_fp)
    gt_rows = read_lm_kbc_jsonl(ground_truth_fp)

    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict:
        # get and normalize the ground truth objects
        gts = []
        for gt in gt_dict[(subj, rel)]:
            gts.append([clean_object(obj) for obj in gt])

        # get and normalize the predictions
        preds = list(set(clean_object(obj) for obj in pred_dict.get((subj, rel), [])))

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f1_score(p, r)

        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1
        })

        # if p > 1.0 or r > 1.0:
        #     print(f"{subj} {rel} {p} {r} {f1} {gts} {preds}")

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def combine_scores_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "p": r["p"],
            "r": r["r"],
            "f1": r["f1"],
        })

    for rel in scores:
        scores[rel] = {
            "p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Precision, Recall and F1-score of predictions"
    )

    parser.add_argument(
        "-p",
        "--predictions",
        type=str,
        default="predictions/gpt3.pred.jsonl",
        required=True,
        help="Path to the predictions file (required)"
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        default="data/dev.jsonl",
        required=True,
        help="Path to the ground truth file (required)"
    )

    args = parser.parse_args()

    scores_per_sr_pair = evaluate_per_sr_pair(args.predictions,
                                              args.ground_truth)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
    }

    print(pd.DataFrame(scores_per_relation).transpose().round(3))

    # Get the model name string
    model = args.predictions.split('/')[-1].split('.')[0]

    print_results(model, args.predictions, args.ground_truth, scores_per_sr_pair, scores_per_relation)


def print_results(model, predictions_fp, ground_truth_fp, scores_per_sr_pair, scores_per_relation):
    scores_per_relation.pop('*** Average ***')

    pred_rows = read_lm_kbc_jsonl(predictions_fp)
    gt_rows = read_lm_kbc_jsonl(ground_truth_fp)

    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    original_stdout = sys.stdout
    output_dir = Path(f'./failure_cases/{model}/')

    if output_dir.exists():
        assert output_dir.is_dir()
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    for relation, average_score in scores_per_relation.items():

        with open(f'./failure_cases/{model}/{relation}.txt', 'w') as f:
            sys.stdout = f
            bad_examples = []
            for instance in scores_per_sr_pair:
                # print(instance.keys())  # dict_keys(['SubjectEntity', 'Relation', 'p', 'r', 'f1'])
                if instance['Relation'] == relation and instance['f1'] <= average_score['f1']:
                    bad_examples.append(instance)

            print(f"{relation} (average f1: {round(average_score['f1'], 3)}): {len(bad_examples)} cases")
            print('\n\n')
            for instance in bad_examples:
                print(f"SubjectEntity: {instance['SubjectEntity']}")
                gt = gt_dict[(instance['SubjectEntity'], instance['Relation'])]
                gt = [x for xs in gt for x in xs]
                gt.sort()
                if gt is not None:
                    gt = [i.lower() for i in gt]
                print(f"Ground Truth: {gt}")
                pr = pred_dict[(instance['SubjectEntity'], instance['Relation'])]
                pr.sort()
                if pr is not None:
                    pr = [i.lower() for i in pr]
                print(f"GPT-3 Prediction: {pr}")
                print('\n')
            sys.stdout = original_stdout  # Reset the standard output to its original value


if __name__ == "__main__":
    main()
