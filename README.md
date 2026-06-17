# qPCR 분자진단 머신러닝 모델별 코드

GUI 없이 모델별 Python 파일 하나씩 실행하는 구성입니다.

## 파일

```text
1-M. Logistic_Regression.py
2-M. Random_Forest.py
3-M. SVM.py
4-M. KNN.py
5-M. Gradient_Boosting.py
```

## 입력

### Excel 한 개

`qPCR_data.xlsx` 안에 다음 시트가 필요합니다.

```text
training
testing
```

### CSV 두 개

각 코드 상단을 다음처럼 바꿉니다.

```python
TRAIN_CSV = SCRIPT_DIR / "training.csv"
TEST_CSV = SCRIPT_DIR / "testing.csv"
```

## 열 구조

```text
Sample,target,Ct_Target,Ct_IC,DeltaCt,RFU,Slope
S001,0,38.1,22.0,16.1,800,0.08
S002,1,25.4,21.9,3.5,9500,0.84
```

- `target=0`: Negative
- `target=1`: Positive
- `Sample`은 선택 사항
- 나머지 숫자 열은 marker로 자동 인식

## 공통 결과

- marker 조합 순위
- 교차검증 ROC-AUC
- testing ROC-AUC
- Accuracy
- Sensitivity
- Specificity
- Precision
- F1
- Average Precision
- ROC curve
- Precision-Recall curve
- Confusion Matrix
- Calibration curve
- cutoff 분석
- permutation importance
- testing 예측 결과
- 학습 모델

로지스틱 회귀는 coefficient와 odds ratio를 추가로 만듭니다.

Random Forest와 Gradient Boosting은 자체 feature importance를 추가로 만듭니다.

## 설치

```bash
python -m pip install pandas numpy scikit-learn matplotlib openpyxl joblib
```

## 실행

```bash
python3 "1-M. Logistic_Regression.py"
python3 "2-M. Random_Forest.py"
python3 "3-M. SVM.py"
python3 "4-M. KNN.py"
python3 "5-M. Gradient_Boosting.py"
```

## marker 조합 설정

```python
RUN_MARKER_COMBINATIONS = True
MAX_COMBINATION_SIZE = 3
```

모든 marker를 한 번에 사용할 때:

```python
RUN_MARKER_COMBINATIONS = False
```
