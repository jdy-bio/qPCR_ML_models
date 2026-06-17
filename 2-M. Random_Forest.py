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
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


SCRIPT_DIR = Path(__file__).resolve().parent

# 사용자가 수정할 부분

# Excel 파일 2개를 사용할 때
TRAIN_FILE = SCRIPT_DIR / "training_data.xlsx"
TEST_FILE = SCRIPT_DIR / "testing_data.xlsx"

# CSV 파일 2개를 사용할 때는 위 Excel 경로를 None으로 바꾸고
TRAIN_CSV = None
TEST_CSV = None

# Excel 파일에서 읽을 시트
# 각 Excel 파일의 첫 번째 시트를 사용하려면 0으로 둡니다.
TRAIN_SHEET = 0
TEST_SHEET = 0

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

MODEL_NAME = "Random_Forest"
OUTPUT_DIR = SCRIPT_DIR / "results_Random_Forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_data():

    using_csv = TRAIN_CSV is not None or TEST_CSV is not None
    using_excel = TRAIN_FILE is not None or TEST_FILE is not None

    if using_csv:
        if TRAIN_CSV is None or TEST_CSV is None:
            raise ValueError("CSV를 사용하려면 지정하세요.")

        if not TRAIN_CSV.exists():
            raise FileNotFoundError(f"학습용 CSV 파일이 없습니다: {TRAIN_CSV}")

        if not TEST_CSV.exists():
            raise FileNotFoundError(f"테스트용 CSV 파일이 없습니다: {TEST_CSV}")

        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)
        return train, test

    if using_excel:
        if TRAIN_FILE is None or TEST_FILE is None:
            raise ValueError("Excel을 사용하려면 지정하세요.")

        if not TRAIN_FILE.exists():
            raise FileNotFoundError(f"학습용 Excel 파일이 없습니다: {TRAIN_FILE}")

        if not TEST_FILE.exists():
            raise FileNotFoundError(f"테스트용 Excel 파일이 없습니다: {TEST_FILE}")

        train = pd.read_excel(TRAIN_FILE, sheet_name=TRAIN_SHEET)
        test = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
        return train, test

    raise ValueError("파일 2개의 경로를 지정하세요.")


def validate_data(train, test):
    """
    학습 데이터에는 target이 반드시 있어야 합니다.
    테스트 데이터의 target은 선택 사항입니다.
    """

    if TARGET_COLUMN not in train.columns:
        raise ValueError(
            f"학습 데이터에 '{TARGET_COLUMN}' 열이 필요합니다."
        )

    excluded = [TARGET_COLUMN]

    if SAMPLE_COLUMN in train.columns:
        excluded.append(SAMPLE_COLUMN)

    features = [column for column in train.columns if column not in excluded]

    if not features:
        raise ValueError("분석할 marker 열이 없습니다.")

    missing_features = [
        column
        for column in features
        if column not in test.columns
    ]

    if missing_features:
        raise ValueError(
            "테스트 데이터에 다음 marker 열이 없습니다: "
            + ", ".join(missing_features)
        )

    train[TARGET_COLUMN] = pd.to_numeric(
        train[TARGET_COLUMN],
        errors="raise",
    ).astype(int)

    train_labels = set(train[TARGET_COLUMN].dropna().unique())

    if not train_labels.issubset({NEGATIVE_LABEL, POSITIVE_LABEL}):
        raise ValueError("target은 0과 1만 사용하세요.")

    if len(train_labels) < 2:
        raise ValueError("양성과 음성이 모두 필요합니다.")

    # marker를 숫자로 변환
    for feature in features:
        train[feature] = pd.to_numeric(train[feature], errors="coerce", )
        test[feature] = pd.to_numeric(test[feature], errors="coerce", )

    if TARGET_COLUMN in test.columns:
        test[TARGET_COLUMN] = pd.to_numeric(test[TARGET_COLUMN], errors="raise", ).astype(int)
        test_labels = set(test[TARGET_COLUMN].dropna().unique())

        if not test_labels.issubset({NEGATIVE_LABEL, POSITIVE_LABEL}):
            raise ValueError("테스트 데이터 target은 0과 1만 사용하세요.")

    return features


def build_model():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_metrics(y_true, probability, cutoff):
    prediction = (probability >= cutoff).astype(int)

    cm = confusion_matrix(y_true, prediction, labels=[NEGATIVE_LABEL, POSITIVE_LABEL], )
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:
        roc_auc = roc_auc_score(y_true, probability)
        average_precision = average_precision_score(y_true, probability)
    else:
        roc_auc = np.nan
        average_precision = np.nan

    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
            pos_label=POSITIVE_LABEL,
        ),
        "F1": f1_score(
            y_true,
            prediction,
            zero_division=0,
            pos_label=POSITIVE_LABEL,
        ),
        "ROC_AUC": roc_auc,
        "Average_Precision": average_precision,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "prediction": prediction,
        "confusion_matrix": cm,
    }


def choose_cutoff(y_true, probability):

    rows = []

    for cutoff in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = calculate_metrics(y_true, probability, cutoff)

        rows.append(
            {
                "Cutoff": cutoff,
                "Sensitivity": metrics["Sensitivity"],
                "Specificity": metrics["Specificity"],
                "Youden_J": (
                    metrics["Sensitivity"]
                    + metrics["Specificity"]
                    - 1
                ),
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
            }
        )

    table = pd.DataFrame(rows)
    eligible = table[table["Sensitivity"] >= TARGET_SENSITIVITY]

    if not eligible.empty:
        selected = eligible.sort_values(
            ["Specificity", "Cutoff"],
            ascending=[False, False],
        ).iloc[0]
        rule = (
            f"Sensitivity >= {TARGET_SENSITIVITY:.2f} 중 "
            "specificity 최대"
        )
    else:
        selected = table.sort_values(["Youden_J", "Cutoff"], ascending=[False, False], ).iloc[0]
        rule = "Youden J 최대"
        
    return table, float(selected["Cutoff"]), rule


def feature_combinations(features):
    if not RUN_MARKER_COMBINATIONS:
        return [tuple(features)]

    maximum = min(MAX_COMBINATION_SIZE, len(features))
    combinations = []

    for size in range(1, maximum + 1):
        combinations.extend(itertools.combinations(features, size))
    return combinations


def save_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{MODEL_NAME} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, )
    plt.close()


def save_test_evaluation(
    test,
    y_test,
    probability,
    metrics,
    selected_cutoff,
    best_features,
    final_model,
):

    report = classification_report(
        y_test,
        metrics["prediction"],
        labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_excel(OUTPUT_DIR / "classification_report.xlsx")

    if len(np.unique(y_test)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_test, probability, pos_label=POSITIVE_LABEL, )

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {metrics['ROC_AUC']:.3f}", )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{MODEL_NAME} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ROC_curve.png", dpi=300, )
        plt.close()

        pd.DataFrame(
            {
                "FPR": fpr,
                "TPR": tpr,
                "Threshold": roc_thresholds,
            }
        ).to_excel(
            OUTPUT_DIR / "ROC_curve_points.xlsx",
            index=False,
        )

        precision_values, recall_values, pr_thresholds = (precision_recall_curve(y_test, probability, pos_label=POSITIVE_LABEL, ))

        plt.figure()
        plt.plot(
            recall_values,
            precision_values,
            label=(
                "AP = "
                f"{metrics['Average_Precision']:.3f}"
            ),
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{MODEL_NAME} Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Precision_Recall_curve.png", dpi=300, )
        plt.close()

        pd.DataFrame(
            {
                "Precision": precision_values[:-1],
                "Recall": recall_values[:-1],
                "Threshold": pr_thresholds,
            }
        ).to_excel(
            OUTPUT_DIR / "Precision_Recall_curve_points.xlsx",
            index=False,
        )

        n_bins = max(2, min(10, len(y_test)),)

        prob_true, prob_pred = calibration_curve(
            y_test,
            probability,
            n_bins=n_bins,
            strategy="quantile",
        )

        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o", label=MODEL_NAME, )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Positive Fraction")
        plt.title(f"{MODEL_NAME} Calibration Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=300, )
        plt.close()

        importance = permutation_importance(
            final_model,
            test[best_features],
            y_test,
            n_repeats=30,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
        )

        importance_df = pd.DataFrame(
            {
                "Feature": best_features,
                "Importance_Mean": (
                    importance.importances_mean
                ),
                "Importance_SD": (
                    importance.importances_std
                ),
            }
        ).sort_values(
            "Importance_Mean",
            ascending=False,
        )

        importance_df.to_excel(OUTPUT_DIR / "permutation_importance.xlsx", index=False, )

        plt.figure()
        plt.barh(
            importance_df["Feature"],
            importance_df["Importance_Mean"],
            xerr=importance_df["Importance_SD"],
        )
        plt.xlabel("Decrease in ROC-AUC")
        plt.ylabel("Marker")
        plt.title(f"{MODEL_NAME} Permutation Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=300, )
        plt.close()

    save_confusion_matrix(metrics["confusion_matrix"])


def main():
    train, test = read_data()

    has_test_target = TARGET_COLUMN in test.columns
    features = validate_data(train, test)
    y_train = train[TARGET_COLUMN]
    class_counts = y_train.value_counts()
    folds = min(CV_FOLDS, int(class_counts.min()),)

    if folds < 2:
        raise ValueError("최소 2개 샘플이 필요합니다.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE, )

    ranking_rows = []
    cv_probabilities = {}

    for combination in feature_combinations(features):
        selected = list(combination)
        model = build_model()

        cv_probability = cross_val_predict(
            model,
            train[selected],
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        cv_metrics = calculate_metrics(
            y_train,
            cv_probability,
            DEFAULT_CUTOFF,
        )

        marker_key = ", ".join(selected)
        cv_probabilities[marker_key] = cv_probability

        ranking_rows.append(
            {
                "Model": MODEL_NAME,
                "Markers": marker_key,
                "Marker_Count": len(selected),
                "CV_ROC_AUC": cv_metrics["ROC_AUC"],
                "CV_Average_Precision": (
                    cv_metrics["Average_Precision"]
                ),
                "CV_Accuracy": cv_metrics["Accuracy"],
                "CV_Sensitivity": cv_metrics["Sensitivity"],
                "CV_Specificity": cv_metrics["Specificity"],
                "CV_Precision": cv_metrics["Precision"],
                "CV_F1": cv_metrics["F1"],
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        [
            "CV_ROC_AUC",
            "CV_Average_Precision",
            "CV_Sensitivity",
            "CV_Specificity",
        ],
        ascending=[False, False, False, False],
    )

    ranking.to_excel(
        OUTPUT_DIR / "marker_combination_ranking.xlsx",
        index=False,
    )

    best_marker_key = ranking.iloc[0]["Markers"]
    best_features = best_marker_key.split(", ")
    best_cv_probability = cv_probabilities[best_marker_key]

    cutoff_table, selected_cutoff, cutoff_rule = (
        choose_cutoff(
            y_train,
            best_cv_probability,
        )
    )

    cutoff_table.to_excel(
        OUTPUT_DIR / "cutoff_table.xlsx",
        index=False,
    )

    plt.figure()
    plt.plot(cutoff_table["Cutoff"], cutoff_table["Sensitivity"], label="Sensitivity", )
    plt.plot(cutoff_table["Cutoff"], cutoff_table["Specificity"], label="Specificity", )
    plt.axvline(selected_cutoff, linestyle="--", label=f"Selected {selected_cutoff:.2f}", )
    plt.xlabel("Cutoff")
    plt.ylabel("Metric")
    plt.title(f"{MODEL_NAME} Cutoff Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cutoff_analysis.png", dpi=300, )
    plt.close()

    final_model = build_model()
    final_model.fit(train[best_features], y_train,)
    probability = final_model.predict_proba(test[best_features])[:, 1]
    prediction = (probability >= selected_cutoff).astype(int)
    predictions = test.copy()

    if SAMPLE_COLUMN not in predictions.columns:
        predictions.insert(
            0,
            SAMPLE_COLUMN,
            [
                f"TEST_{index + 1:03d}"
                for index in range(len(predictions))
            ],
        )

    predictions["Positive_Probability"] = np.round(probability, 5, )
    predictions["Selected_Cutoff"] = selected_cutoff
    predictions["Prediction"] = prediction
    predictions["Prediction_Label"] = (
        predictions["Prediction"].map(
            {
                NEGATIVE_LABEL: "Negative",
                POSITIVE_LABEL: "Positive",
            }
        )
    )

    predictions.to_excel(OUTPUT_DIR / "predicted_results.xlsx", index=False, )

    # Random Forest 자체 변수 중요도
    fitted_model = final_model.named_steps["model"]
    feature_names = (final_model.named_steps["imputer"].get_feature_names_out(best_features))

    native_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": fitted_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False, )

    native_importance_df.to_excel(OUTPUT_DIR / "random_forest_feature_importance.xlsx", index=False, )

    joblib.dump(
        {
            "model": final_model,
            "features": best_features,
            "model_name": MODEL_NAME,
            "selected_cutoff": selected_cutoff,
            "target_column": TARGET_COLUMN,
            "sample_column": SAMPLE_COLUMN,
            "negative_label": NEGATIVE_LABEL,
            "positive_label": POSITIVE_LABEL,
        },
        OUTPUT_DIR / "trained_model.joblib",
    )

    summary = {
        "Model": MODEL_NAME,
        "Mode": (
            "Evaluation"
            if has_test_target
            else "Prediction_Only"
        ),
        "Best_Markers": best_features,
        "Training_Samples": int(len(train)),
        "Testing_Samples": int(len(test)),
        "Selected_Cutoff": float(selected_cutoff),
        "Cutoff_Rule": cutoff_rule,
        "CV_ROC_AUC": float(
            ranking.iloc[0]["CV_ROC_AUC"]
        ),
        "CV_Average_Precision": float(
            ranking.iloc[0]["CV_Average_Precision"]
        ),
    }

    if has_test_target:
        y_test = test[TARGET_COLUMN]

        test_metrics = calculate_metrics(
            y_test,
            probability,
            selected_cutoff,
        )

        save_test_evaluation(
            test=test,
            y_test=y_test,
            probability=probability,
            metrics=test_metrics,
            selected_cutoff=selected_cutoff,
            best_features=best_features,
            final_model=final_model,
        )

        summary.update(
            {
                "Test_ROC_AUC": (
                    None
                    if np.isnan(test_metrics["ROC_AUC"])
                    else float(test_metrics["ROC_AUC"])
                ),
                "Test_Average_Precision": (
                    None
                    if np.isnan(
                        test_metrics["Average_Precision"]
                    )
                    else float(
                        test_metrics["Average_Precision"]
                    )
                ),
                "Accuracy": float(
                    test_metrics["Accuracy"]
                ),
                "Sensitivity": float(
                    test_metrics["Sensitivity"]
                ),
                "Specificity": float(
                    test_metrics["Specificity"]
                ),
                "Precision": float(
                    test_metrics["Precision"]
                ),
                "F1": float(test_metrics["F1"]),
                "TN": test_metrics["TN"],
                "FP": test_metrics["FP"],
                "FN": test_metrics["FN"],
                "TP": test_metrics["TP"],
            }
        )

        print("테스트 데이터의 target을 이용해 성능평가를 수행했습니다.")
        print(
            f"정확도: {test_metrics['Accuracy']:.4f}"
        )
        print(
            f"민감도: {test_metrics['Sensitivity']:.4f}"
        )
        print(
            f"특이도: {test_metrics['Specificity']:.4f}"
        )

        if not np.isnan(test_metrics["ROC_AUC"]):
            print(
                "ROC-AUC: "
                f"{test_metrics['ROC_AUC']:.4f}"
            )
        else:
            print(
                "테스트 데이터에 한 종류의 target만 있어 "
                "ROC-AUC는 계산하지 않았습니다."
            )

    else:
        print(
            "테스트 데이터에 target이 없어 "
            "미지 샘플 예측만 수행했습니다."
        )

    with open(
        OUTPUT_DIR / "analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame([summary]).to_excel(
        OUTPUT_DIR / "analysis_summary.xlsx",
        index=False,
    )

    print(f"{MODEL_NAME} 분석 완료")
    print(f"최적 marker: {', '.join(best_features)}")
    print(f"선택 cutoff: {selected_cutoff:.2f}")
    print(f"결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"분석 실패: {error}")
        raise
