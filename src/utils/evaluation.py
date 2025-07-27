import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, entropy
from sklearn.preprocessing import LabelEncoder
from dython.nominal import associations
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def load_all_csvs_to_datasetdict(base_path: str, delimiter: str = ",") -> DatasetDict:
    """
    Scan the given base_path for all .csv files, load them into HuggingFace Dataset objects,
    and return a DatasetDict where the key is the filename without '.csv'.
    """
    dataset_dict = {}

    # Ensure the path exists
    if not os.path.isdir(base_path):
        raise ValueError(f"Base path does not exist or is not a directory: {base_path}")

    # Iterate over all CSV files in the directory
    for fname in os.listdir(base_path):
        if fname.lower().endswith(".csv"):
            csv_path = os.path.join(base_path, fname)
            # Load CSV with pandas
            df = pd.read_csv(csv_path, delimiter=delimiter)
            # Key is filename without .csv
            key = os.path.splitext(fname)[0]
            # Convert to Hugging Face Dataset
            dataset_dict[key] = Dataset.from_pandas(df)

    # Wrap into DatasetDict
    ds_dict = DatasetDict(dataset_dict)
    return ds_dict

def jensen_shannon_divergence(p, q):
    """Compute Jensen-Shannon divergence between two probability distributions."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # normalize
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))

def compute_correlation_difference(real_df, synth_df):
    # Get association results
    corr_real = associations(real_df, nominal_columns='all', plot=False)
    corr_synth = associations(synth_df, nominal_columns='all', plot=False)

    # If dicts, extract the matrix
    if isinstance(corr_real, dict):
        corr_real = corr_real.get('corr', corr_real)
    if isinstance(corr_synth, dict):
        corr_synth = corr_synth.get('corr', corr_synth)

    # Now both should be DataFrames
    corr_diff = (corr_real - corr_synth).abs().mean().mean()
    return corr_diff

def distance_to_closest_record(real, synth):
    dists = cdist(synth, real, metric='euclidean')
    closest = np.min(dists, axis=1)
    return np.percentile(closest, 5)


def nearest_neighbor_distance_ratio(real, synth):
    dists = cdist(synth, real, metric='euclidean')
    sorted_dists = np.sort(dists, axis=1)
    ratios = sorted_dists[:, 0] / (sorted_dists[:, 1] + 1e-9)
    return np.percentile(ratios, 5)


def evaluate_ml_utility_with_splits(real_train, real_test, synth_train, target_col):
    X_train_real = real_train.drop(columns=[target_col])
    y_train_real = real_train[target_col]
    X_test = real_test.drop(columns=[target_col])
    y_test = real_test[target_col]
    X_train_synth = synth_train.drop(columns=[target_col])
    y_train_synth = synth_train[target_col]  # should correspond in size/labels

    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MLP": MLPClassifier(max_iter=500)
    }

    results = {}
    for name, model in models.items():
        # Real train
        model.fit(X_train_real, y_train_real)
        y_pred_real = model.predict(X_test)
        y_proba_real = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        metrics_real = {
            "accuracy": accuracy_score(y_test, y_pred_real),
            "f1": f1_score(y_test, y_pred_real, average="weighted"),
            "auc": roc_auc_score(y_test, y_proba_real) if y_proba_real is not None else None
        }

        # Synthetic train
        model.fit(X_train_synth, y_train_synth)
        y_pred_synth = model.predict(X_test)
        y_proba_synth = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        metrics_synth = {
            "accuracy": accuracy_score(y_test, y_pred_synth),
            "f1": f1_score(y_test, y_pred_synth, average="weighted"),
            "auc": roc_auc_score(y_test, y_proba_synth) if y_proba_synth is not None else None
        }

        results[name] = {"train_real": metrics_real, "train_synth": metrics_synth}

    return results

def evaluate(general_cfg: Dict[str, Any], data_cfg: List[Dict[str, Any]]) -> None:
    overall_results = []

    for spec in data_cfg:
        dataset_name: str = spec["dataset"]
        base_path: str = spec["dataset_path"]
        target_col: str = spec["target_column"]

        # Load real and synthetic datasets
        ds = load_all_csvs_to_datasetdict(base_path)

        # Check that the dataset contains train and test splits
        if not all(k in ds for k in ["train", "test"]):
            raise ValueError(f"{dataset_name} must contain at least 'train' and 'test' splits.")

        # Real train and test data
        real_train = ds["train"].to_pandas()
        real_test = ds["test"].to_pandas()

        # Loop through every synthetic method (every split that is not train/test)
        synth_splits = [k for k in ds.keys() if k not in ["train", "test"]]
        if not synth_splits:
            logger.warning(f"No synthetic splits found in {dataset_name}. Skipping.")
            continue

        # Identify categorical columns BEFORE encoding
        categorical_cols = [col for col in real_train.columns if real_train[col].dtype == "object"]

        for synth_method in synth_splits:

            synth_train = ds[synth_method].to_pandas()

            # Encode categorical columns
            for col in categorical_cols:
                le = LabelEncoder()
                all_values = pd.concat([real_train[col], real_test[col], synth_train[col].astype(str)])
                le.fit(all_values.astype(str))
                real_train[col] = le.transform(real_train[col].astype(str))
                real_test[col] = le.transform(real_test[col].astype(str))
                synth_train[col] = le.transform(synth_train[col].astype(str).fillna(""))

            # ML utility
            ml_results = evaluate_ml_utility_with_splits(real_train, real_test, synth_train, target_col)

            # Statistical similarity
            jsd_results = {}
            for col in real_train.columns:
                if col == target_col:
                    continue
                if real_train[col].dtype == "int" or real_train[col].dtype == "object":
                    p_real = real_train[col].value_counts(normalize=True)
                    p_synth = synth_train[col].value_counts(normalize=True)
                    combined_index = p_real.index.union(p_synth.index)
                    p = p_real.reindex(combined_index, fill_value=0)
                    q = p_synth.reindex(combined_index, fill_value=0)
                    jsd_results[col] = jensen_shannon_divergence(p, q)

            wd_results = {}
            for col in real_train.columns:
                if col == target_col:
                    continue
                if np.issubdtype(real_train[col].dtype, np.number):
                    wd_results[col] = wasserstein_distance(real_train[col], synth_train[col])

            corr_diff = compute_correlation_difference(
                real_train.drop(columns=[target_col]),
                synth_train.drop(columns=[target_col])
            )

            # Privacy
            X_real = real_train.drop(columns=[target_col]).to_numpy()
            X_synth = synth_train.drop(columns=[target_col]).to_numpy()
            dcr = distance_to_closest_record(X_real, X_synth)
            nndr = nearest_neighbor_distance_ratio(X_real, X_synth)

            overall_results.append({
                "dataset": dataset_name,
                "synth_method": synth_method,
                "ml_results": ml_results,
                "statistical_similarity_jsd": jsd_results,
                "statistical_similarity_wd": wd_results,
                "statistical_similarity_corr_diff": corr_diff,
                "privacy_preservability_DCR_5th_percentile": dcr,
                "privacy_preservability_NNDR_5th_percentile": nndr
            })

    # Flatten the nested ML results
    flat_rows = []
    for entry in overall_results:
        dataset = entry["dataset"]
        synth_method = entry["synth_method"]
        for model_name, metrics in entry["ml_results"].items():
            flat_rows.append({
                "dataset": dataset,
                "synth_method": synth_method,
                "model": model_name,
                "accuracy_real": metrics["train_real"]["accuracy"],
                "f1_real": metrics["train_real"]["f1"],
                "auc_real": metrics["train_real"]["auc"] if metrics["train_real"]["auc"] is not None else -1,
                "accuracy_synth": metrics["train_synth"]["accuracy"],
                "f1_synth": metrics["train_synth"]["f1"],
                "auc_synth": metrics["train_synth"]["auc"] if metrics["train_synth"]["auc"] is not None else -1,
                "jsd": entry["statistical_similarity_jsd"],
                "wasserstein": entry["statistical_similarity_wd"],
                "corr_diff": entry["statistical_similarity_corr_diff"],
                "DCR_5th": entry["privacy_preservability_DCR_5th_percentile"],
                "NNDR_5th": entry["privacy_preservability_NNDR_5th_percentile"]
            })

    # Save as CSV
    output_dir = general_cfg.get("output_dir", "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    logger.info(f"Evaluation saved to: {csv_path}")

def evaluate_beer(general_cfg: Dict[str, Any], data_cfg: List[Dict[str, Any]]) -> None:
    overall_results = []

    for spec in data_cfg:
        dataset_name: str = spec["dataset"]
        base_path: str = spec["dataset_path"]

        # Load real and synthetic CSVs
        real_df = pd.read_csv(f"{base_path}/real.csv", delimiter=";")
        synth_df = pd.read_csv(f"{base_path}/synthetic.csv", delimiter=";")

        real_df = real_df.dropna()
        synth_df = synth_df.dropna()

        # Identify categorical columns BEFORE encoding
        categorical_cols = [col for col in real_df.columns if real_df[col].dtype == "object"]

        # Encode categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            all_values = pd.concat([real_df[col], synth_df[col].astype(str)])
            le.fit(all_values.astype(str))
            real_df[col] = le.transform(real_df[col].astype(str))
            synth_df[col] = le.transform(synth_df[col].astype(str).fillna(""))

        # Compute JSD only for those categorical columns
        jsd_results = {}
        for col in categorical_cols:
            p_real = real_df[col].value_counts(normalize=True)
            p_synth = synth_df[col].value_counts(normalize=True)
            combined_index = p_real.index.union(p_synth.index)
            p = p_real.reindex(combined_index, fill_value=0)
            q = p_synth.reindex(combined_index, fill_value=0)
            jsd_results[col] = jensen_shannon_divergence(p, q)

        wd_results = {}
        for col in real_df.columns:
            if np.issubdtype(real_df[col].dtype, np.number):
                # treat as numeric
                wd_results[col] = wasserstein_distance(real_df[col], synth_df[col])

        corr_diff = compute_correlation_difference(real_df, synth_df)

        # Privacy metrics
        X_real = real_df.to_numpy()
        X_synth = synth_df.to_numpy()
        dcr = distance_to_closest_record(X_real, X_synth)
        nndr = nearest_neighbor_distance_ratio(X_real, X_synth)

        overall_results.append({
            "statistical_similarity_jsd": jsd_results,
            "statistical_similarity_wd": wd_results,
            "statistical_similarity_corr_diff": corr_diff,
            "privacy_preservability_DCR_5th_percentile": dcr,
            "privacy_preservability_NNDR_5th_percentile": nndr
        })

    # Flatten into CSV rows
    flat_rows = []
    for entry in overall_results:
        flat_rows.append({
            # convert dicts to a plain string representation (no JSON escaping)
            "jsd": str(entry["statistical_similarity_jsd"]),
            "wasserstein": str(entry["statistical_similarity_wd"]),
            "corr_diff": entry["statistical_similarity_corr_diff"],
            "DCR_5th": entry["privacy_preservability_DCR_5th_percentile"],
            "NNDR_5th": entry["privacy_preservability_NNDR_5th_percentile"]
        })

    output_dir = general_cfg.get("output_dir", "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    logger.info(f"Evaluation saved to: {csv_path}")