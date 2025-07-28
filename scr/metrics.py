import numpy as np
from collections import defaultdict

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    def get_entities(seq):
        entities = []
        start = None
        label = None
        for i, tag in enumerate(seq):
            if tag.startswith("B-"):
                if start is not None:
                    entities.append((start, i, label))
                start = i
                label = tag[2:]
            elif tag.startswith("I-") and start is not None and tag[2:] == label:
                continue
            else:
                if start is not None:
                    entities.append((start, i, label))
                    start = None
                    label = None
        if start is not None:
            entities.append((start, len(seq), label))
        return entities

    def overlaps(e1, e2):
        return e1[2] == e2[2] and max(e1[0], e2[0]) < min(e1[1], e2[1])

    exact_tp = defaultdict(int)
    partial_tp = defaultdict(int)
    total_pred_entities = defaultdict(int)
    total_true_entities = defaultdict(int)
    exact_fp = defaultdict(int)

    for pred_seq, true_seq in zip(true_predictions, true_labels):
        pred_ents = get_entities(pred_seq)
        true_ents = get_entities(true_seq)

        for ent in true_ents:
            total_true_entities[ent[2]] += 1
        for ent in pred_ents:
            total_pred_entities[ent[2]] += 1

        matched_pred = set()
        matched_true = set()

        for i, true_ent in enumerate(true_ents):
            for j, pred_ent in enumerate(pred_ents):
                if true_ent == pred_ent:
                    exact_tp[true_ent[2]] += 1
                    matched_pred.add(j)
                    matched_true.add(i)
                    break

        for i, true_ent in enumerate(true_ents):
            if i in matched_true:
                continue
            for j, pred_ent in enumerate(pred_ents):
                if j in matched_pred:
                    continue
                if overlaps(true_ent, pred_ent):
                    partial_tp[true_ent[2]] += 1
                    matched_pred.add(j)
                    matched_true.add(i)
                    break

        for pred_ent in pred_ents:
            has_exact = any(pred_ent == true_ent for true_ent in true_ents)
            if not has_exact:
                exact_fp[pred_ent[2]] += 1

    total_true_all = sum(total_true_entities.values())
    total_pred_all = sum(total_pred_entities.values())
    exact_tp_all = sum(exact_tp.values())
    partial_tp_all = sum(partial_tp.values())
    exact_fp_all = sum(exact_fp.values())

    exact_precision = exact_tp_all / (exact_tp_all + exact_fp_all) if (exact_tp_all + exact_fp_all) else 0
    exact_recall = exact_tp_all / total_true_all if total_true_all else 0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if (exact_precision + exact_recall) else 0

    partial_tp_total = exact_tp_all + partial_tp_all
    partial_precision = partial_tp_total / total_pred_all if total_pred_all else 0
    partial_recall = partial_tp_total / total_true_all if total_true_all else 0
    partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall) if (partial_precision + partial_recall) else 0

    metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "exact_precision": exact_precision,
        "exact_recall": exact_recall,
        "exact_f1": exact_f1,
        "partial_precision": partial_precision,
        "partial_recall": partial_recall,
        "partial_f1": partial_f1,
    }

    all_entities = set(total_true_entities.keys()).union(total_pred_entities.keys())
    for entity in all_entities:
        etp = exact_tp[entity]
        efp = exact_fp[entity]
        epred = total_pred_entities[entity]
        etrue = total_true_entities[entity]

        e_prec = etp / (etp + efp) if (etp + efp) else 0
        e_rec = etp / etrue if etrue else 0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) else 0

        ptp = etp + partial_tp[entity]
        p_prec = ptp / epred if epred else 0
        p_rec = ptp / etrue if etrue else 0
        p_f1 = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) else 0

        metrics[f"{entity}_exact_precision"] = e_prec
        metrics[f"{entity}_exact_recall"] = e_rec
        metrics[f"{entity}_exact_f1"] = e_f1
        metrics[f"{entity}_partial_precision"] = p_prec
        metrics[f"{entity}_partial_recall"] = p_rec
        metrics[f"{entity}_partial_f1"] = p_f1

    for entity, scores in results.items():
        if entity.startswith("overall"):
            continue
        if isinstance(scores, dict) and "precision" in scores:
            metrics[f"{entity}_precision"] = scores["precision"]
            metrics[f"{entity}_recall"] = scores["recall"]
            metrics[f"{entity}_f1"] = scores["f1"]

    return metrics