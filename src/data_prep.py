import argparse
import math
import os
from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = " ".join(text.split())
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "it's": "it is",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    return text


def build_prompt(question: str) -> str:
    # Stronger instruction to bias answers toward Rwanda-specific, practical guidance
    instruction = (
        "You are an agriculture assistant for smallholder farmers in Rwanda. "
        "Answer briefly and specifically for Rwanda conditions (mention seasons, local varieties, or practical steps). "
        "If timing is asked, consider Season A (Sept–Dec) and Season B (Mar–Jun). "
        "If unsure, advise what to check locally (rainfall, soil pH, or an extension officer)."
    )
    return f"{instruction} Question: {question} Answer:"


def synthesize_rwanda_qa(n: int = 100) -> Dataset:
    topics = [
        {"crop": "maize", "tips": [
            "plant maize at the beginning of the long rains (Season B: Mar–Jun) for better yield",
            "use improved varieties like ZM607 or SC513 adapted to Rwandan soils",
            "monitor for fall armyworm; handpick or use biocontrol early",
            "apply organic compost and a balanced NPK fertilizer after 3 weeks",
        ]},
        {"crop": "beans", "tips": [
            "plant climbing beans on stakes for better yield on hillsides",
            "use improved varieties like MAC44 or RWV1129",
            "plant at the onset of rains (Season A or B)",
            "rotate beans with maize or cassava to prevent soil fatigue",
        ]},
        {"crop": "potatoes", "tips": [
            "use certified seed potatoes (Kinigi, Victoria) and avoid replanting old tubers",
            "hill soil around plants twice before flowering",
            "rotate with beans or maize to reduce late blight risk",
            "avoid waterlogging; ensure drainage on terraces",
        ]},
        {"crop": "cassava", "tips": [
            "use clean, disease-free cuttings to avoid mosaic disease",
            "plant during the onset of rains and maintain 1x1m spacing",
            "apply organic manure before planting",
        ]},
        {"crop": "banana", "tips": [
            "dig deep holes (60cm x 60cm) and mix topsoil with compost before planting",
            "mulch and maintain suckers regularly",
            "apply manure twice a year for good bunch size",
        ]},
        {"crop": "coffee", "tips": [
            "plant coffee seedlings at the start of the short rains (Season A: Sept–Dec)",
            "prune and mulch to conserve moisture",
            "use shade trees and control pests like coffee berry borer",
        ]},
        {"crop": "tea", "tips": [
            "keep proper spacing and pluck regularly for quality leaves",
            "apply nitrogen fertilizer in small, frequent doses",
            "control soil acidity by applying lime when pH is below 4.5",
        ]},
    ]
    seasons = [("season a", "sept-dec"), ("season b", "mar-jun")]
    soil_general = [
        "add compost and mulch to improve soil fertility",
        "construct terraces or contour bunds on slopes to reduce erosion",
        "apply lime if the soil pH is below 5.5",
        "practice crop rotation to maintain soil health",
        "test your soil every two seasons through Rwanda Agriculture Board (RAB)",
    ]
    soil_conditions: List[Tuple[str, str]] = [
        ("sandy soils", "add organic matter frequently and mulch to retain moisture"),
        ("clay soils", "improve drainage with raised beds and add compost"),
        ("acidic soils", "apply agricultural lime after a soil test"),
        ("loamy soils", "maintain with compost and avoid over-tillage"),
    ]
    invalid_diseases = ["malaria", "flu", "covid-19"]

    questions: List[str] = []
    answers: List[str] = []

    # 1) Crop and season planting advice
    for t in topics:
        for tip in t["tips"]:
            for sname, srange in seasons:
                q = f"when should i plant {t['crop']} in rwanda {sname}?"
                a = f"in rwanda {sname} ({srange}), {tip}. adjust to local rainfall."
                questions.append(normalize_text(q))
                answers.append(normalize_text(a))

        # 2) Soil management advice per crop
        for soil_type, advice in soil_conditions:
            q = f"is {soil_type} good for growing {t['crop']} in rwanda?"
            a = f"{advice}. {t['crop']} can grow in {soil_type} if managed properly."
            questions.append(normalize_text(q))
            answers.append(normalize_text(a))

        # General soil practices per crop
        for practice in soil_general:
            q2 = f"how do i manage soil for {t['crop']} on rwanda hills?"
            a2 = f"{practice}. also add compost and mulch."
            questions.append(normalize_text(q2))
            answers.append(normalize_text(a2))

    # 3) Soil-based crop suggestions
    for soil_type, advice in soil_conditions:
        q = f"which crop is best to grow in {soil_type} in rwanda?"
        if "sandy" in soil_type:
            a = "maize, cassava, and groundnuts grow well in sandy soils with organic matter."
        elif "clay" in soil_type:
            a = "beans, rice, and vegetables do well in clay soils if drainage is improved."
        elif "acidic" in soil_type:
            a = "lime the soil first, then grow maize or beans for better yield."
        else:
            a = "most crops including maize and beans thrive in loamy soils."
        a += f" also remember to {advice}."
        questions.append(normalize_text(q))
        answers.append(normalize_text(a))

    # 4) Invalid disease awareness
    for disease in invalid_diseases:
        for crop in ["maize", "beans", "cassava"]:
            q = f"does {crop} get {disease}?"
            a = (
                f"no, {disease} affects humans, not crops. for {crop}, watch out for real plant "
                f"diseases like leaf blight or mosaic."
            )
            questions.append(normalize_text(q))
            answers.append(normalize_text(a))

    # 5) Cooperative and general questions
    while len(questions) < n:
        questions.append(normalize_text("how do cooperatives help farmers in rwanda?"))
        answers.append(normalize_text("they improve market access, training, and fair prices for farmers."))

    questions = questions[:n]
    answers = answers[:n]
    df = pd.DataFrame({"question": questions, "answer": answers})
    return Dataset.from_pandas(df, preserve_index=False)


def synthesize_general_agri_qa(n: int = 10000) -> Dataset:
    crops = ["wheat", "rice", "soybean", "barley", "sorghum", "millet", "groundnut", "sunflower"]
    top_qs = [
        "what is the best planting depth for {crop}?",
        "how much water does {crop} need per week?",
        "what fertilizer is recommended for {crop}?",
        "how to control common pests in {crop}?",
        "when to harvest {crop}?",
    ]
    top_as = [
        "plant at 2-5 cm depending on soil moisture",
        "aim for consistent moisture; avoid waterlogging",
        "apply balanced npk based on soil test",
        "use integrated pest management and crop rotation",
        "harvest at physiological maturity for best quality",
    ]
    questions: List[str] = []
    answers: List[str] = []
    for crop in crops:
        for q_tmpl, a in zip(top_qs, top_as):
            q = q_tmpl.format(crop=crop)
            questions.append(normalize_text(q))
            answers.append(normalize_text(a))
    base_len = len(questions)
    idx = 0
    while len(questions) < n:
        q = questions[idx % base_len]
        a = answers[idx % base_len]
        questions.append(q)
        answers.append(a)
        idx += 1
    questions = questions[:n]
    answers = answers[:n]
    df = pd.DataFrame({"question": questions, "answer": answers})
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_input_length: int, max_target_length: int) -> Dataset:
    def _map_fn(batch: Dict[str, List[str]]):
        prompts = [build_prompt(q) for q in batch["question"]]
        model_inputs = tokenizer(
            prompts,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["answer"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    cols = ["question", "answer"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in cols])
    return dataset.map(_map_fn, batched=True, remove_columns=cols)


def train_val_test_split(dataset: Dataset, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    dataset = dataset.shuffle(seed=seed)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_ds = dataset.select(range(0, n_train))
    val_ds = dataset.select(range(n_train, n_train + n_val))
    test_ds = dataset.select(range(n_train + n_val, n))
    return train_ds, val_ds, test_ds


def save_splits(splits: Dict[str, Dataset], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, ds in splits.items():
        ds.save_to_disk(os.path.join(out_dir, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwanda_csv", type=str, default="")
    parser.add_argument("--general_csv", type=str, default="")
    parser.add_argument("--generate_rwanda", type=int, default=400)   # you can increase
    parser.add_argument("--generate_general", type=int, default=4000)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--upsample_limit", type=int, default=10,
                        help="Maximum repeat multiplier when upsampling Rwanda dataset")
    args = parser.parse_args()

    datasets_list: List[Dataset] = []
    if args.rwanda_csv and os.path.exists(args.rwanda_csv):
        df = pd.read_csv(args.rwanda_csv)
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must contain 'question' and 'answer' columns")
        df["question"] = df["question"].apply(normalize_text)
        df["answer"] = df["answer"].apply(normalize_text)
        datasets_list.append(Dataset.from_pandas(df, preserve_index=False))
    else:
        datasets_list.append(synthesize_rwanda_qa(max(1, args.generate_rwanda)))

    if args.general_csv and os.path.exists(args.general_csv):
        df2 = pd.read_csv(args.general_csv)
        if "question" not in df2.columns or "answer" not in df2.columns:
            raise ValueError("CSV must contain 'question' and 'answer' columns")
        df2["question"] = df2["question"].apply(normalize_text)
        df2["answer"] = df2["answer"].apply(normalize_text)
        datasets_list.append(Dataset.from_pandas(df2, preserve_index=False))
    else:
        datasets_list.append(synthesize_general_agri_qa(max(1, args.generate_general)))

    # -----------------------
    # Upsample Rwanda dataset to reduce imbalance
    # -----------------------
    if len(datasets_list) == 1:
        full = datasets_list[0]
        print("Only one dataset provided; skipping upsampling.")
    else:
        rwanda_ds = datasets_list[0]
        general_ds = datasets_list[1]
        r_len = len(rwanda_ds)
        g_len = len(general_ds)
        if r_len == 0:
            full = concatenate_datasets([rwanda_ds, general_ds]).shuffle(seed=42)
            print("Rwanda dataset empty; using concatenation without upsampling.")
        else:
            mult = max(1, math.ceil(g_len / r_len))
            mult = min(mult, args.upsample_limit)
            if mult > 1:
                print(f"Upsampling Rwanda dataset: repeating it {mult} times (capped at {args.upsample_limit}).")
                rwanda_upsampled = concatenate_datasets([rwanda_ds] * mult)
            else:
                rwanda_upsampled = rwanda_ds
                print("Rwanda dataset already similar size to general dataset; no upsampling needed.")
            full = concatenate_datasets([rwanda_upsampled, general_ds]).shuffle(seed=42)
            print(f"After upsampling, full dataset size: {len(full)} (Rwanda ~{len(rwanda_upsampled)}, General ~{len(general_ds)})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tok = tokenize_dataset(full, tokenizer, args.max_input_length, args.max_target_length)
    train_ds, val_ds, test_ds = train_val_test_split(tok)
    save_splits({"train": train_ds, "val": val_ds, "test": test_ds}, args.output_dir)
    print(f"✅ Saved processed datasets to: {args.output_dir}")


if __name__ == "__main__":
    main()
