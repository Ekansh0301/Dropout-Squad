"""
reporter.py — Evaluation script for Hybrid Player Intent Classifier
Automatically finds dataset and ensures correct column headers.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("intent_reporter")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def auto_discover_paths():
    """Auto-detect model, dataset, and output directories."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models", "intent_classifier", "final")
    output_dir = os.path.join(base_dir, "results", "intent_eval")

    # Try both likely dataset locations
    dataset_candidates = [
        os.path.join(base_dir, "data", "intent_test.csv"),
        os.path.join(base_dir, "data", "processed", "hybrid_player_data.csv")
    ]
    dataset_path = None
    for path in dataset_candidates:
        if os.path.exists(path):
            dataset_path = path
            logger.info(f"✅ Found dataset: {path}")
            break
    if dataset_path is None:
        raise FileNotFoundError("No dataset found in ./data/intent_test.csv or ./data/processed/hybrid_player_data.csv")

    return model_dir, dataset_path, output_dir


def load_model(model_dir):
    logger.info(f"Loading model from: {model_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def load_and_validate_dataset(path):
    """Load dataset and ensure correct column names."""
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Detect and fix column naming variations
    columns = [c.lower() for c in df.columns]
    df.columns = columns

    if "text" not in df.columns:
        raise ValueError(f"Dataset missing required 'text' column: found {df.columns}")

    # Handle possible label column names
    if "intent" in df.columns:
        df.rename(columns={"intent": "label"}, inplace=True)
    elif "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' or 'intent' column.")

    logger.info(f"Dataset loaded with {len(df)} rows and columns: {df.columns.tolist()}")
    return df


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Intent Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_reliability_diagram(y_true, y_probs, save_path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_intent_classifier(model_dir, dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_model(model_dir)
    df = load_and_validate_dataset(dataset_path)

    # Handle int or string keys robustly
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    label_list = [id2label[i] for i in sorted(id2label.keys())]


    # Encode dataset
    inputs = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    labels = torch.tensor([label2id.get(lbl, 2) for lbl in df["label"]])  # default to DIALOGUE if mismatch

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        labels = labels.to("cuda")

    # -----------------------------------------------------------------
    # Batched evaluation to avoid GPU OOM
    # -----------------------------------------------------------------
    from tqdm import trange

    batch_size = 64  # adjust to fit your GPU; 32–128 are typical
    all_probs, all_preds = [], []

    logger.info(f"Evaluating {len(df)} samples in batches of {batch_size}...")
    with torch.no_grad():
        for i in trange(0, len(df), batch_size, desc="Evaluating"):
            batch_inputs = {
                k: v[i:i + batch_size] for k, v in inputs.items()
            }
            outputs = model(**batch_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)


    y_true = labels.cpu().numpy()
    y_pred = preds
    y_conf = probs[np.arange(len(y_true)), y_pred]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=label_list, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, label_list, os.path.join(output_dir, "confusion_matrix.png"))

    # ROC-AUC (macro)
    try:
        y_true_onehot = np.eye(len(label_list))[y_true]
        auc = roc_auc_score(y_true_onehot, probs, multi_class="ovr")
        logger.info(f"Macro ROC-AUC: {auc:.4f}")
    except Exception as e:
        logger.warning(f"ROC-AUC not computed: {e}")
        auc = None

    # Reliability diagram
    plot_reliability_diagram((y_true == y_pred).astype(int), y_conf,
                             os.path.join(output_dir, "reliability_diagram.png"))

    # Save predictions
    df_out = df.copy()
    # handle int keys gracefully
    df_out["pred_label"] = [id2label.get(i, id2label.get(str(i), "UNKNOWN")) for i in y_pred]

    df_out["confidence"] = y_conf
    df_out.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    logger.info(f"Results saved to: {output_dir}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Player Intent Classifier")
    parser.add_argument("--model_dir", default=None, help="Path to model dir")
    parser.add_argument("--dataset", default=None, help="Path to dataset CSV")
    parser.add_argument("--output_dir", default=None, help="Path to output directory")
    args = parser.parse_args()

    # Auto-detect paths
    model_dir, dataset_path, output_dir = auto_discover_paths()
    if args.model_dir:
        model_dir = args.model_dir
    if args.dataset:
        dataset_path = args.dataset
    if args.output_dir:
        output_dir = args.output_dir

    results = evaluate_intent_classifier(model_dir, dataset_path, output_dir)

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        if v is not None:
            print(f"{k:>10s}: {v:.4f}")
