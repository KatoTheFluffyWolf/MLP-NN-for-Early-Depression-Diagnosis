"""
Leakage-safe training script for the Early Depression Diagnosis repository.

Outputs:
- artifacts/depression_detection.keras
- artifacts/preprocessor.joblib
- artifacts/feature_config.json
- artifacts/test_metrics.json
- artifacts/test_predictions.csv
- artifacts/confusion_matrix.csv
- artifacts/training_history.csv
- artifacts/training_loss.png
- artifacts/training_accuracy.png
- artifacts/model_architecture.png (when Graphviz/pydot are available)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DATASET_HANDLE = "adilshamim8/student-depression-dataset"
DATA_FILE = "student_depression_dataset.csv"
TARGET_COLUMN = "Depression"

NUMERICAL_FEATURES = [
    "Age",
    "CGPA",
    "Academic Pressure",
    "Study Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]

OUTPUT_DIR = Path("artifacts")
MODEL_PATH = OUTPUT_DIR / "depression_detection.keras"
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.joblib"
CONFIG_PATH = OUTPUT_DIR / "feature_config.json"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_reproducible_seed(seed: int = SEED) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_layer_name(text: str) -> str:
    """Convert a feature label into a valid, readable Keras layer name."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    return cleaned or "feature"


def json_ready(value: Any) -> Any:
    """Convert NumPy values into JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_dataset() -> pd.DataFrame:
    """Download and load the Kaggle dataset as a pandas DataFrame."""
    print("Loading dataset...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        DATA_FILE,
    )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("kagglehub did not return a pandas DataFrame.")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select required columns and clean numeric, categorical, and target data."""
    required_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    cleaned = df[required_columns].copy()

    # Convert numeric fields safely.
    for column in NUMERICAL_FEATURES:
        cleaned[column] = (
            cleaned[column]
            .replace("?", np.nan)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Preserve missing categorical values as an explicit category.
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = (
            cleaned[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
        cleaned.loc[cleaned[column] == "", column] = "Unknown"

    # Convert the target to 0/1 while supporting either numeric or text labels.
    if pd.api.types.is_numeric_dtype(cleaned[TARGET_COLUMN]):
        cleaned[TARGET_COLUMN] = pd.to_numeric(
            cleaned[TARGET_COLUMN], errors="coerce"
        )
    else:
        target_mapping = {
            "0": 0,
            "1": 1,
            "no": 0,
            "yes": 1,
            "false": 0,
            "true": 1,
        }
        cleaned[TARGET_COLUMN] = (
            cleaned[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

    cleaned = cleaned.dropna(
        subset=NUMERICAL_FEATURES + [TARGET_COLUMN]
    ).reset_index(drop=True)

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(np.int32)

    invalid_targets = set(cleaned[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(
            f"{TARGET_COLUMN!r} must contain only binary labels 0 and 1. "
            f"Found: {sorted(invalid_targets)}"
        )

    if cleaned[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The cleaned target must contain both classes.")

    return cleaned


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified 70/15/15 train, validation, and test splits.
    """
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
        stratify=df[TARGET_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
        stratify=temporary_df[TARGET_COLUMN],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def fit_preprocessors(
    train_df: pd.DataFrame,
) -> tuple[StandardScaler, OrdinalEncoder]:
    """
    Fit preprocessing only on the training split to avoid data leakage.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERICAL_FEATURES])

    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )
    categorical_encoder.fit(train_df[CATEGORICAL_FEATURES])

    return scaler, categorical_encoder


def transform_split(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    categorical_encoder: OrdinalEncoder,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the fitted training preprocessors to one dataset split.

    Category indices are shifted by +1:
    - 0 is reserved for unseen categories.
    - Known training categories use indices 1..N.
    """
    numerical_array = scaler.transform(
        frame[NUMERICAL_FEATURES]
    ).astype(np.float32)

    categorical_matrix = categorical_encoder.transform(
        frame[CATEGORICAL_FEATURES]
    ).astype(np.int32)

    categorical_matrix = categorical_matrix + 1

    categorical_inputs = [
        categorical_matrix[:, index : index + 1]
        for index in range(len(CATEGORICAL_FEATURES))
    ]

    model_inputs = [numerical_array] + categorical_inputs
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

    return model_inputs, targets, numerical_array, categorical_matrix


def build_model(
    categorical_encoder: OrdinalEncoder,
) -> keras.Model:
    """Build an MLP with embeddings for categorical features."""
    numerical_input = keras.Input(
        shape=(len(NUMERICAL_FEATURES),),
        dtype=tf.float32,
        name="numerical_input",
    )

    model_inputs: list[keras.KerasTensor] = [numerical_input]
    embedded_features: list[keras.KerasTensor] = []

    for index, feature_name in enumerate(CATEGORICAL_FEATURES):
        layer_name = safe_layer_name(feature_name)

        categorical_input = keras.Input(
            shape=(1,),
            dtype=tf.int32,
            name=f"{layer_name}_input",
        )

        # Known categories are 1..N and unseen categories are 0.
        vocabulary_size = len(categorical_encoder.categories_[index]) + 1
        embedding_dimension = min(
            8,
            max(2, int(np.ceil(np.sqrt(vocabulary_size)))),
        )

        embedding = layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dimension,
            name=f"{layer_name}_embedding",
        )(categorical_input)

        embedding = layers.Flatten(
            name=f"{layer_name}_flatten"
        )(embedding)

        model_inputs.append(categorical_input)
        embedded_features.append(embedding)

    combined_input = layers.Concatenate(name="combined_features")(
        embedded_features + [numerical_input]
    )

    x = layers.Dense(64, activation="relu", name="dense_64")(combined_input)
    x = layers.Dropout(0.20, name="dropout_20")(x)
    x = layers.Dense(32, activation="relu", name="dense_32")(x)
    x = layers.Dropout(0.10, name="dropout_10")(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="depression_probability",
    )(x)

    model = keras.Model(
        inputs=model_inputs,
        outputs=output,
        name="student_depression_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def save_preprocessing_artifacts(
    scaler: StandardScaler,
    categorical_encoder: OrdinalEncoder,
) -> None:
    """Save fitted preprocessing objects and readable feature metadata."""
    joblib.dump(
        {
            "scaler": scaler,
            "categorical_encoder": categorical_encoder,
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_column": TARGET_COLUMN,
            "category_index_offset": 1,
        },
        PREPROCESSOR_PATH,
    )

    config = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "category_index_offset": 1,
        "categorical_input_names": [
            f"{safe_layer_name(name)}_input"
            for name in CATEGORICAL_FEATURES
        ],
        "categories": {
            feature_name: categorical_encoder.categories_[index].tolist()
            for index, feature_name in enumerate(CATEGORICAL_FEATURES)
        },
    }

    CONFIG_PATH.write_text(
        json.dumps(json_ready(config), indent=2),
        encoding="utf-8",
    )


def evaluate_model(
    model: keras.Model,
    test_inputs: list[np.ndarray],
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate on the untouched test split and save detailed outputs."""
    keras_results = model.evaluate(
        test_inputs,
        y_test,
        verbose=0,
        return_dict=True,
    )

    probabilities = model.predict(
        test_inputs,
        verbose=0,
    ).reshape(-1)

    predictions = (probabilities >= 0.5).astype(np.int32)

    metrics: dict[str, Any] = {
        "loss": float(keras_results["loss"]),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    confusion = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        confusion,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    prediction_output = test_df.copy()
    prediction_output["predicted_probability"] = probabilities
    prediction_output["predicted_class"] = predictions
    prediction_output.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2),
        encoding="utf-8",
    )

    return metrics


def save_training_outputs(
    model: keras.Model,
    history: keras.callbacks.History,
) -> None:
    """Save the model, training history, plots, and architecture diagram."""
    model.save(MODEL_PATH)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    if {"loss", "val_loss"}.issubset(history_df.columns):
        plt.figure(figsize=(7, 4.5))
        plt.plot(history_df["loss"], label="Training loss")
        plt.plot(history_df["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Binary cross-entropy")
        plt.title("Training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=200)
        plt.close()

    if {"accuracy", "val_accuracy"}.issubset(history_df.columns):
        plt.figure(figsize=(7, 4.5))
        plt.plot(history_df["accuracy"], label="Training accuracy")
        plt.plot(history_df["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_accuracy.png", dpi=200)
        plt.close()

    try:
        keras.utils.plot_model(
            model,
            to_file=str(OUTPUT_DIR / "model_architecture.png"),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
        )
    except Exception as exc:
        print(
            "Model architecture image was not generated. "
            "Install Graphviz and pydot if you need it. "
            f"Details: {exc}"
        )


def print_split_summary(
    name: str,
    frame: pd.DataFrame,
) -> None:
    distribution = (
        frame[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_dict()
    )
    print(f"{name}: {len(frame):,} rows | class distribution: {distribution}")


def main() -> None:
    set_reproducible_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset()
    df = clean_dataset(raw_df)

    train_df, validation_df, test_df = split_dataset(df)

    print_split_summary("Train", train_df)
    print_split_summary("Validation", validation_df)
    print_split_summary("Test", test_df)

    scaler, categorical_encoder = fit_preprocessors(train_df)

    train_inputs, y_train, _, _ = transform_split(
        train_df,
        scaler,
        categorical_encoder,
    )
    validation_inputs, y_validation, _, _ = transform_split(
        validation_df,
        scaler,
        categorical_encoder,
    )
    test_inputs, y_test, _, _ = transform_split(
        test_df,
        scaler,
        categorical_encoder,
    )

    save_preprocessing_artifacts(
        scaler,
        categorical_encoder,
    )

    model = build_model(categorical_encoder)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(validation_inputs, y_validation),
        batch_size=32,
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_outputs(model, history)

    metrics = evaluate_model(
        model,
        test_inputs,
        y_test,
        test_df,
    )

    print("\nUntouched test-set results")
    print("-" * 32)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nSaved files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()
