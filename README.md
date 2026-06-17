# qPCR 분자진단 머신러닝 모델별 코드

## 파일

```text
1-M. Logistic_Regression.py
2-M. Random_Forest.py
3-M. SVM.py
4-M. KNN.py
5-M. Gradient_Boosting.py
```

## 입력

### CSV, Excel 필요

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

## 각 모델들을 특징

```text
1. Logistic Regression

각 마커가 진단 결과에 미치는 영향을 계수와 오즈비로 계산합니다.

마커 값이 증가할수록 양성 가능성이 증가하는지 확인 가능
마커별 영향 방향과 크기를 설명하기 쉬움
마커 사이 관계가 비교적 단순할 때 적합
결과 해석과 보고서 작성에 유리

2. Random Forest

여러 결정트리가 각 마커의 기준값을 반복적으로 나누어 양성과 음성을 분류합니다.

마커와 진단 결과 사이의 비선형 관계 분석 가능
여러 마커가 함께 작용하는 상호작용 탐색 가능
마커 중요도 제공
일부 이상치와 노이즈에 비교적 강함

3. SVM

여러 마커 조합을 이용해 양성과 음성을 가장 잘 구분하는 경계를 만듭니다.

직선으로 구분하기 어려운 복잡한 마커 패턴 분석 가능
적은 샘플과 여러 마커를 사용하는 데이터에 유리할 수 있음
마커별 영향 방향을 직접 설명하기는 어려움
분류 성능 비교용 모델로 적합

4. KNN

새 샘플과 마커 값이 가장 비슷한 학습 샘플들을 찾아 판정합니다.

유사한 Ct 또는 ΔCt 패턴을 가진 샘플끼리 같은 군으로 분류
모델 구조가 단순함
마커 수가 많거나 이상치가 많으면 성능이 저하될 수 있음
데이터 스케일 차이에 민감하므로 표준화가 필요

5. Gradient Boosting

작은 결정트리를 순차적으로 만들면서 이전 모델이 틀린 샘플을 반복적으로 보완합니다.

복잡한 마커 관계와 비선형 패턴 분석 가능
마커 간 상호작용 반영 가능
높은 예측 성능을 낼 가능성이 있음
데이터가 적으면 과적합될 수 있음
