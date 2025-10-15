import argparse
import math
import os
import json
from typing import Optional

import tensorflow as tf
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    TFAutoModelForSeq2SeqLM,
    create_optimizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_generate_tokens", type=int, default=64)
    return parser.parse_args()


def load_splits(processed_dir: str):
    train_path = os.path.join(processed_dir, "train")
    val_path = os.path.join(processed_dir, "val")
    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        raise FileNotFoundError(
            f"Expected processed splits at '{train_path}' and '{val_path}'. Run data prep first."
        )
    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path)
    return train_ds, val_ds


def build_tf_dataset(hf_dataset, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    # Datasets are already padded to max length during preprocessing
    tf_ds = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=None,
    )
    return tf_ds


def create_tf_optimizer(
    name: str,
    learning_rate: float,
    warmup_ratio: float,
    steps_per_epoch: int,
    epochs: int,
) -> tf.keras.optimizers.Optimizer:
    if name == "adamw":
        total_steps = steps_per_epoch * max(1, epochs)
        warmup_steps = int(warmup_ratio * total_steps)
        optimizer, _ = create_optimizer(
            init_lr=learning_rate,
            num_warmup_steps=warmup_steps,
            num_train_steps=total_steps,
            weight_decay_rate=0.01,
        )
        return optimizer
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds, val_ds = load_splits(args.processed_dir)

    train_tf = build_tf_dataset(train_ds, batch_size=args.batch_size, shuffle=True)
    val_tf = build_tf_dataset(val_ds, batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    optimizer = create_tf_optimizer(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
    )

    model.compile(optimizer=optimizer)

    # Prepare checkpoint/backup paths
    ckpt_dir = os.path.join(args.output_dir, "checkpoint")
    backup_dir = os.path.join(args.output_dir, "backup_resume")

    # If checkpoint exists, try to resume weights before training
    if os.path.exists(ckpt_dir):
        try:
            model.load_weights(ckpt_dir)
            print(f"Loaded existing checkpoint weights from: {ckpt_dir}")
        except Exception as e:
            print(f"Warning: could not load checkpoint from {ckpt_dir}: {e}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_dir,
            monitor="val_loss",
            save_best_only=False,  # save every epoch so we can resume even without best val
            save_weights_only=True,
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
    ]

    model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training configuration and a minimal history summary
    history_path = os.path.join(args.output_dir, "training_summary.json")
    config = {
        "processed_dir": args.processed_dir,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "warmup_ratio": args.warmup_ratio,
        "max_generate_tokens": args.max_generate_tokens,
    }
    try:
        # Keras fit returns History with history attribute; capture basic metrics if available
        # Note: Above, we didn't assign the History, so run one small epoch summary by evaluate
        metrics = model.evaluate(val_tf, return_dict=True, verbose=0)
    except Exception:
        metrics = {}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "val_metrics": metrics}, f, indent=2)

    # Quick sanity generate on a batch for verification
    for batch in val_tf.take(1):
        inputs = {"input_ids": batch[0]["input_ids"], "attention_mask": batch[0]["attention_mask"]}
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_generate_tokens,
            num_beams=4,
        )
        _ = tokenizer.batch_decode(generated, skip_special_tokens=True)
        break

    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()



