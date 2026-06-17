from pathlib import Path
import itertools
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

SCRIPT_DIR = Path(__file__).resolve().parent

# =========================================================
# 사용자가 수정할 부분
# =========================================================

# Excel 하나 사용: training/testing 시트 필요
INPUT_FILE = SCRIPT_DIR / "qPCR_data.xlsx"

# CSV 두 개 사용 시 경로를 지정하고, Excel 사용 시 None 유지
TRAIN_CSV = None
TEST_CSV = None

TRAIN_SHEET = "training"
TEST_SHEET = "testing"

TARGET_COLUMN = "target"
SAMPLE_COLUMN = "Sample"

NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1

RUN_MARKER_COMBINATIONS = True
MAX_COMBINATION_SIZE = 3

DEFAULT_CUTOFF = 0.50
TARGET_SENSITIVITY = 0.95

CV_FOLDS = 5
RANDOM_STATE = 42

MODEL_NAME = "Gradient_Boosting"
OUTPUT_DIR = SCRIPT_DIR / f"results_Gradient_Boosting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_data():
    if TRAIN_CSV is not None and TEST_CSV is not None:
        return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    if INPUT_FILE.suffix.lower() in [".xlsx", ".xls"]:
        train = pd.read_excel(INPUT_FILE, sheet_name=TRAIN_SHEET)
        test = pd.read_excel(INPUT_FILE, sheet_name=TEST_SHEET)
        return train, test

    raise ValueError(
        "Excel은 INPUT_FILE을 사용하고, CSV는 TRAIN_CSV와 TEST_CSV를 지정하세요."
    )


def validate_data(train, test):
    if TARGET_COLUMN not in train.columns or TARGET_COLUMN not in test.columns:
        raise ValueError(f"training/testing 데이터에 {TARGET_COLUMN} 열이 필요합니다.")

    excluded = [TARGET_COLUMN]
    if SAMPLE_COLUMN in train.columns:
        excluded.append(SAMPLE_COLUMN)

    features = [c for c in train.columns if c not in excluded]

    if not features:
        raise ValueError("분석할 marker 열이 없습니다.")

    missing = [c for c in features if c not in test.columns]
    if missing:
        raise ValueError("testing 데이터에 다음 열이 없습니다: " + ", ".join(missing))

    for df in [train, test]:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)

        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

        values = set(df[TARGET_COLUMN].unique())
        if not values.issubset({NEGATIVE_LABEL, POSITIVE_LABEL}):
            raise ValueError("target은 0과 1만 사용하세요.")

        if len(values) < 2:
            raise ValueError("training/testing 각각에 양성·음성이 모두 필요합니다.")

    return features


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ))
    ])


def calculate_metrics(y_true, probability, cutoff):
    prediction = (probability >= cutoff).astype(int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probability),
        "Average_Precision": average_precision_score(y_true, probability),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "prediction": prediction,
        "confusion_matrix": cm,
    }


def choose_cutoff(y_true, probability):
    rows = []

    for cutoff in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = calculate_metrics(y_true, probability, cutoff)

        rows.append({
            "Cutoff": cutoff,
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Youden_J": metrics["Sensitivity"] + metrics["Specificity"] - 1,
            "TN": metrics["TN"], "FP": metrics["FP"],
            "FN": metrics["FN"], "TP": metrics["TP"],
        })

    table = pd.DataFrame(rows)
    eligible = table[table["Sensitivity"] >= TARGET_SENSITIVITY]

    if not eligible.empty:
        selected = eligible.sort_values(
            ["Specificity", "Cutoff"],
            ascending=[False, False]
        ).iloc[0]
        rule = f"Sensitivity >= {TARGET_SENSITIVITY:.2f} 중 specificity 최대"
    else:
        selected = table.sort_values("Youden_J", ascending=False).iloc[0]
        rule = "Youden J 최대"

    return table, float(selected["Cutoff"]), rule


def feature_combinations(features):
    if not RUN_MARKER_COMBINATIONS:
        return [tuple(features)]

    maximum = min(MAX_COMBINATION_SIZE, len(features))
    result = []

    for size in range(1, maximum + 1):
        result.extend(itertools.combinations(features, size))

    return result


def save_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(cm[row, col]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Gradient_Boosting Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def main():
    train, test = read_data()
    features = validate_data(train, test)

    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    class_counts = y_train.value_counts()
    folds = min(CV_FOLDS, int(class_counts.min()))

    if folds < 2:
        raise ValueError("교차검증을 위해 각 군에 최소 2개 샘플이 필요합니다.")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    ranking_rows = []

    for combination in feature_combinations(features):
        selected = list(combination)
        model = build_model()

        cv_probability = cross_val_predict(
            model,
            train[selected],
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        model.fit(train[selected], y_train)
        test_probability = model.predict_proba(test[selected])[:, 1]

        metrics = calculate_metrics(y_test, test_probability, DEFAULT_CUTOFF)

        ranking_rows.append({
            "Model": MODEL_NAME,
            "Markers": ", ".join(selected),
            "Marker_Count": len(selected),
            "CV_ROC_AUC": roc_auc_score(y_train, cv_probability),
            "Test_ROC_AUC": metrics["ROC_AUC"],
            "Average_Precision": metrics["Average_Precision"],
            "Accuracy": metrics["Accuracy"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Precision": metrics["Precision"],
            "F1": metrics["F1"],
        })

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["CV_ROC_AUC", "Test_ROC_AUC", "Sensitivity", "Specificity"],
        ascending=[False, False, False, False]
    )

    ranking.to_excel(
        OUTPUT_DIR / "marker_combination_ranking.xlsx",
        index=False
    )

    best_features = ranking.iloc[0]["Markers"].split(", ")

    final_model = build_model()
    final_model.fit(train[best_features], y_train)

    probability = final_model.predict_proba(test[best_features])[:, 1]

    cutoff_table, selected_cutoff, cutoff_rule = choose_cutoff(
        y_test,
        probability
    )

    selected_metrics = calculate_metrics(
        y_test,
        probability,
        selected_cutoff
    )

    cutoff_table.to_excel(
        OUTPUT_DIR / "cutoff_table.xlsx",
        index=False
    )

    predictions = test.copy()

    if SAMPLE_COLUMN not in predictions.columns:
        predictions.insert(
            0,
            SAMPLE_COLUMN,
            [f"TEST_{i + 1:03d}" for i in range(len(predictions))]
        )

    predictions["Positive_Probability"] = np.round(probability, 5)
    predictions["Selected_Cutoff"] = selected_cutoff
    predictions["Prediction"] = selected_metrics["prediction"]
    predictions["Prediction_Label"] = predictions["Prediction"].map({
        0: "Negative",
        1: "Positive",
    })

    predictions.to_excel(
        OUTPUT_DIR / "predicted_results.xlsx",
        index=False
    )

    report = classification_report(
        y_test,
        selected_metrics["prediction"],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).T.to_excel(
        OUTPUT_DIR / "classification_report.xlsx"
    )

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, probability)

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {selected_metrics['ROC_AUC']:.3f}"
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Gradient_Boosting ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ROC_curve.png", dpi=300)
    plt.close()

    pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": roc_thresholds,
    }).to_excel(
        OUTPUT_DIR / "ROC_curve_points.xlsx",
        index=False
    )

    # Precision-Recall curve
    precision_values, recall_values, _ = precision_recall_curve(
        y_test,
        probability
    )

    plt.figure()
    plt.plot(
        recall_values,
        precision_values,
        label=f"AP = {selected_metrics['Average_Precision']:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Gradient_Boosting Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Precision_Recall_curve.png", dpi=300)
    plt.close()

    save_confusion_matrix(
        selected_metrics["confusion_matrix"]
    )

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_test,
        probability,
        n_bins=min(10, len(y_test)),
        strategy="quantile"
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label=MODEL_NAME)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title(f"Gradient_Boosting Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=300)
    plt.close()

    # Cutoff plot
    plt.figure()
    plt.plot(
        cutoff_table["Cutoff"],
        cutoff_table["Sensitivity"],
        label="Sensitivity"
    )
    plt.plot(
        cutoff_table["Cutoff"],
        cutoff_table["Specificity"],
        label="Specificity"
    )
    plt.axvline(
        selected_cutoff,
        linestyle="--",
        label=f"Selected {selected_cutoff:.2f}"
    )
    plt.xlabel("Cutoff")
    plt.ylabel("Metric")
    plt.title(f"Gradient_Boosting Cutoff Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cutoff_analysis.png", dpi=300)
    plt.close()

    # Permutation importance
    importance = permutation_importance(
        final_model,
        test[best_features],
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="roc_auc"
    )

    importance_df = pd.DataFrame({
        "Feature": best_features,
        "Importance_Mean": importance.importances_mean,
        "Importance_SD": importance.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    importance_df.to_excel(
        OUTPUT_DIR / "permutation_importance.xlsx",
        index=False
    )

    plt.figure()
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance_Mean"],
        xerr=importance_df["Importance_SD"]
    )
    plt.xlabel("Decrease in ROC-AUC")
    plt.ylabel("Marker")
    plt.title(f"Gradient_Boosting Permutation Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=300)
    plt.close()

    fitted = final_model.named_steps["model"]
    names = final_model.named_steps["imputer"].get_feature_names_out(best_features)

    pd.DataFrame({
        "Feature": names,
        "Importance": fitted.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False
    ).to_excel(
        OUTPUT_DIR / "gradient_boosting_feature_importance.xlsx",
        index=False
    )


    joblib.dump(
        {
            "model": final_model,
            "features": best_features,
            "model_name": MODEL_NAME,
            "selected_cutoff": selected_cutoff,
        },
        OUTPUT_DIR / "trained_model.joblib"
    )

    summary = {
        "Model": MODEL_NAME,
        "Best_Markers": best_features,
        "Training_Samples": int(len(train)),
        "Testing_Samples": int(len(test)),
        "Selected_Cutoff": selected_cutoff,
        "Cutoff_Rule": cutoff_rule,
        "CV_ROC_AUC": float(ranking.iloc[0]["CV_ROC_AUC"]),
        "Test_ROC_AUC": float(selected_metrics["ROC_AUC"]),
        "Average_Precision": float(selected_metrics["Average_Precision"]),
        "Accuracy": float(selected_metrics["Accuracy"]),
        "Sensitivity": float(selected_metrics["Sensitivity"]),
        "Specificity": float(selected_metrics["Specificity"]),
        "Precision": float(selected_metrics["Precision"]),
        "F1": float(selected_metrics["F1"]),
        "TN": selected_metrics["TN"],
        "FP": selected_metrics["FP"],
        "FN": selected_metrics["FN"],
        "TP": selected_metrics["TP"],
    }

    with open(
        OUTPUT_DIR / "analysis_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    pd.DataFrame([summary]).to_excel(
        OUTPUT_DIR / "analysis_summary.xlsx",
        index=False
    )

    print(f"Gradient_Boosting 분석 완료")
    print(f"최적 marker: {', '.join(best_features)}")
    print(f"ROC-AUC: {selected_metrics['ROC_AUC']:.4f}")
    print(f"민감도: {selected_metrics['Sensitivity']:.4f}")
    print(f"특이도: {selected_metrics['Specificity']:.4f}")
    print(f"결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"분석 실패: {error}")
        raise
