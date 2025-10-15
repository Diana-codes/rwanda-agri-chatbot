import argparse
import math
import os
from typing import List, Tuple

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

import sacrebleu
from rouge_score import rouge_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--max_generate_tokens", type=int, default=64)
    parser.add_argument("--report_perplexity", action="store_true")
    return parser.parse_args()


def sample_dataset(ds, k: int):
    k = min(k, len(ds))
    return ds.select(range(k))


def compute_bleu(references: List[str], predictions: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def compute_rouge(references: List[str], predictions: List[str]):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return float(np.mean(r1)), float(np.mean(r2)), float(np.mean(rL))


def compute_token_f1(references: List[str], predictions: List[str]) -> Tuple[float, float, float]:
    precisions, recalls, f1s = [], [], []
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.strip().split()
        pred_tokens = pred.strip().split()
        if len(ref_tokens) == 0 and len(pred_tokens) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        if len(pred_tokens) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        ref_counts = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        match = 0
        for t in pred_tokens:
            if ref_counts.get(t, 0) > 0:
                match += 1
                ref_counts[t] -= 1
        precision = match / max(1, len(pred_tokens))
        recall = match / max(1, len(ref_tokens))
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def compute_perplexity(model: TFAutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, subset) -> float:
    # Average negative log-likelihood over tokens, then exponentiate
    total_loss = 0.0
    total_count = 0
    batch_inputs = []
    batch_labels = []
    batch_masks = []
    batch_size = 8
    for example in subset:
        input_ids = np.array(example["input_ids"], dtype=np.int32)
        attention_mask = np.array(example.get("attention_mask"), dtype=np.int32)
        labels = np.array(example["labels"], dtype=np.int32)
        batch_inputs.append((input_ids, attention_mask))
        batch_labels.append(labels)
        # Count non -100 label tokens
        batch_masks.append((labels != -100).astype(np.int32))
        if len(batch_inputs) == batch_size:
            inputs = {
                "input_ids": np.stack([x[0] for x in batch_inputs], axis=0),
                "attention_mask": np.stack([x[1] for x in batch_inputs], axis=0),
                "labels": np.stack(batch_labels, axis=0),
            }
            outputs = model(**inputs, training=False)
            loss = float(outputs.loss.numpy())
            count = int(np.sum(np.stack(batch_masks, axis=0)))
            total_loss += loss * count
            total_count += count
            batch_inputs, batch_labels, batch_masks = [], [], []
    if batch_inputs:
        inputs = {
            "input_ids": np.stack([x[0] for x in batch_inputs], axis=0),
            "attention_mask": np.stack([x[1] for x in batch_inputs], axis=0),
            "labels": np.stack(batch_labels, axis=0),
        }
        outputs = model(**inputs, training=False)
        loss = float(outputs.loss.numpy())
        count = int(np.sum(np.stack(batch_masks, axis=0)))
        total_loss += loss * count
        total_count += count
    if total_count == 0:
        return float("inf")
    avg_nll = total_loss / total_count
    return math.exp(avg_nll)


def main():
    args = parse_args()

    test_path = os.path.join(args.processed_dir, "test")
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Expected test split at '{test_path}'. Run data prep first.")
    test_ds = load_from_disk(test_path)
    subset = sample_dataset(test_ds, args.num_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    # We need original questions and answers for metrics, so rely on detokenization
    # Note: During preprocessing, inputs were prompts and labels are token ids.
    references: List[str] = []
    predictions: List[str] = []

    for example in subset:
        input_ids = example["input_ids"]
        attention_mask = example.get("attention_mask")
        input_ids_tensor = np.array([input_ids])
        attention_mask_tensor = np.array([attention_mask]) if attention_mask is not None else None
        generated = model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=args.max_generate_tokens,
            num_beams=4,
        )
        pred_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        # Reconstruct reference text from labels (ignore -100 if present)
        label_ids = example["labels"]
        label_ids = [tok for tok in label_ids if tok != -100]
        ref_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        references.append(ref_text)
        predictions.append(pred_text)

    bleu = compute_bleu(references, predictions)
    r1, r2, rL = compute_rouge(references, predictions)
    p_tok, r_tok, f1_tok = compute_token_f1(references, predictions)

    print("Evaluation Report")
    print(f"BLEU: {bleu:.2f}")
    print(f"ROUGE-1 F1: {r1:.3f}")
    print(f"ROUGE-2 F1: {r2:.3f}")
    print(f"ROUGE-L F1: {rL:.3f}")
    print(f"Token-level Precision: {p_tok:.3f}")
    print(f"Token-level Recall: {r_tok:.3f}")
    print(f"Token-level F1: {f1_tok:.3f}")

    if args.report_perplexity:
        ppl = compute_perplexity(model, tokenizer, subset)
        print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()


