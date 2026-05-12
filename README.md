# DLcompetition

# 프로젝트 개요
- 배경: 현대 고객들의 금융 행동은 '직업, 대출 종류, 상환 행태, 신용 혼합도' 등 수많은 변수들이 비선형적으로 얽혀 있어, 기존 모델로는 변수 간의 복잡한 상호작용을 파악하는 데 한계가 존재합니다. 이로 인해 우량 고객(Good)을 놓치거나, 잠재적 부실 고객(Poor)을 조기에 탐지하지 못하는 리스크 비용이 지속적으로 발생
- 목표: 고객의 다양한 금융/비금융 데이터를 융합하여 정교한 신용 점수(Credit Score) 등급을 예측합니다.
- 기간: 2026년 5월 12일

# 사용 데이터
- 데이터 출처: Kaggle Credit score classification Datasest
- 주요 변수: Annual_Income, Outstanding_Debt, Interest_Rate, Delay_from_due_date
- 가변 변수: DTI_Ratio, EMI_to_Income_Ratio, Financial_Reserve, Delay_Score

# 기술 스택
- 데이터 전처리: Pandas, Numpy, Scikit-learn(Preprocessing), Matplotlib, Seaborn
- 모델링: TabNet, TabTransformer

# 데이터 전처리
1. 결측치 확인 - 결측치 없음
2. 불필요한 변수 제거 - ID, Customer_ID, Name, SSN
3. 범주형 변수 인코딩 - Occupation, Type_of_Loan, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour

# EDA
## 1. y(Credit Score)클래스 비율 확인
<img width="709" height="469" alt="image" src="https://github.com/user-attachments/assets/72f915df-c089-44b6-831b-d76bf6833067" />

- y(Credit Score)의 비율 확인 결과, Standard 등급이 53174로 가장 높은 수치를 보이고, Poor 등급이 28998, Good 등급이 17828로 가장 낮은 수치를 기록했다. -> 클래스 불균형

## 2. 수치형 변수 간 상관관계 히트맵
<img width="1264" height="990" alt="image" src="https://github.com/user-attachments/assets/a36ffd97-6a90-4c73-a6f2-cb19d801d19a" />

- 수치형 변수 간의 히트맵 확인 결과, 높은 상관관계를 띄는 변수들이 존재했다.
- Annual_Income, Monthly_Inhand_Salary와 Amount_invested_monthly사이 상관계수가 둘다 0.81이다. -> 개인의 연간 소득 및 월별 기본급여가 높을 수록 월별 투자금액이 같이 상승한다.

- Outstanding_Debt와 Credit_History_Age의 상관계수가 -0.63으로 가장 작은 음의 상관계수를 보이고 있다. -> 신용 거래 기간이 늘면 미지급 잔액이 주는 경우 많은 것으로 보인다.

## 3. 주요 수치형 변수 등급별 분포 확인
<img width="1589" height="990" alt="image" src="https://github.com/user-attachments/assets/943580a8-1ae4-4d45-aca2-b95b22310275" />

- 숫자 0(Good 등급)의 경우, 연봉이 가장 높고, 빚, 연체 일수 매우 낮다. 우량 고객이다.
- 숫자 1(Poor 등급)의 경우, 연봉이 가장 낮은 편이고, 빚, 이자율, 연체 일수가 높다. 위험 고객이다.
- 숫자 2(Standard 등급)의 경우, 모든 지표가 0~1사이에 위치해있다.

## 4. 파생 변수 등급별 분포 확인
<img width="1590" height="989" alt="image" src="https://github.com/user-attachments/assets/44fb88c5-caa4-4daf-ac5e-8e04bad9f5f5" />

- DTI_Ratio (소득 대비 부채 비율)
<br>0(Good)등급은 바닥에 붙어있고, 1(Poor)등급은 박스가 가장 위에 있다.
- EMI_to_Income_Ratio (월 소득 대비 고정지출 비중)
<br>1(Poor)등급의 고객들은 나머지 두 고객들에 비해 월급 대비 고정지출이 높다. -> 당장 사용할 현금이 부족하다.
- Financial_Reserve (잉여 금융 자산)
<br>0(Good)등급의 고객들이 통잔 잔고, 투자금을 합친 '여윳돈'이 높다. 하지만 2(Standard)등급의 고객들과 큰 차이는 없어보인다.
- Delay_Score (종합 연체 위험 지수)
<br>0(Good)등급의 고객들은 바닥에 붙어있는 반면, 1(Poor)등급의 고객의 경우 박스 자체가 매우 넓고 높게 분포되어 있다.
<br>위험 및 불량 고객을 솎아내는데 중요한 역할을 할 것임을 보여준다.

# 모델링
## 1. TabNet
<img width="592" height="342" alt="image" src="https://github.com/user-attachments/assets/2c30b20d-8bb3-41cc-bd5e-96d1829e7959" />

- TabNet 모델의 가장 큰 성과는 데이터 비율이 가장 적었던(약 17,000건) 'Good' 클래스의 재현율(Recall)을 무려 0.90까지 끌어올렸다는 점입니다. 이는 실제 우량 고객 10명 중 9명을 놓치지 않고 정확히 식별해 냈음을 의미합니다.
- 아쉬운 점: Train Accuracy(0.89)와 Test Accuracy(0.81) 사이의 격차가 존재하여 다소 과적합(Overfitting) 경향이 있으며, 'Good' 클래스의 정밀도(Precision)가 0.71로 상대적으로 낮아 다수 클래스(Standard)를 Good으로 오인하는 경우(False Positive)가 일부 발생했습니다.

## 2. TabTransformer 
<img width="581" height="298" alt="image" src="https://github.com/user-attachments/assets/0f8fcdd0-8fb7-45e1-8d33-57a31f0c3322" />

- 상세 분석: TabTransformer는 TabNet과 전체 정확도는 비슷하지만, 클래스 간의 균형 잡힌 예측력을 보여주었습니다. 'Good' 클래스의 정밀도(Precision)가 0.74로 상승했고, 'Standard' 클래스의 재현율 역시 0.76에서 0.78로 개선되었습니다.
- 기술적 원인: 자연어 처리의 트랜스포머(Transformer) 아키텍처를 차용하여, 범주형 변수 간의 복잡하고 다차원적인 상호작용(Interaction)을 깊이 있게 학습함으로써 특정 클래스에 과도하게 편향되지 않고 안정적인 분류 경계를 형성했습니다.

# 비즈니스 관점의 모델 활용 전략
1. VIP 타겟 마케팅 및 고객 유치 목적 ➔ TabNet 채택
- 마케팅 캠페인에서는 단 한 명의 잠재적 우량 고객(Good)도 놓치지 않는 것이 중요합니다. 오분류된 고객에게 혜택이 조금 더 가더라도, 진짜 우량 고객 90%를 확보할 수 있는 TabNet이 공격적인 비즈니스 확장에 유리합니다.

2. 리스크 관리 및 보수적인 대출 심사 목적 ➔ TabTransformer 채택
- 대출 심사에서는 일반 고객(Standard)을 우량 고객(Good)으로 잘못 판단(False Positive)하여 금리 우대를 해주면 회사의 재무적 손실로 이어집니다. 따라서 정밀도(Precision)가 더 높고 오판율이 적어 밸런스가 좋은 TabTransformer가 리스크를 방어하는 데 적합합니다.
