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
1. y(Credit Score)클래스 비율 확인
<img width="709" height="469" alt="image" src="https://github.com/user-attachments/assets/72f915df-c089-44b6-831b-d76bf6833067" />

- y(Credit Score)의 비율 확인 결과, Standard 등급이 53174로 가장 높은 수치를 보이고, Poor 등급이 28998, Good 등급이 17828로 가장 낮은 수치를 기록했다. -> 클래스 불균형

2. 수치형 변수 간 상관관계 히트맵
<img width="1264" height="990" alt="image" src="https://github.com/user-attachments/assets/a36ffd97-6a90-4c73-a6f2-cb19d801d19a" />

- 수치형 변수 간의 히트맵 확인 결과, 높은 상관관계를 띄는 변수들이 존재했다.
- Annual_Income, Monthly_Inhand_Salary와 Amount_invested_monthly사이 상관계수가 둘다 0.81이다. -> 개인의 연간 소득 및 월별 기본급여가 높을 수록 월별 투자금액이 같이 상승한다.

- Outstanding_Debt와 Credit_History_Age의 상관계수가 -0.63으로 가장 작은 음의 상관계수를 보이고 있다. -> 신용 거래 기간이 늘면 미지급 잔액이 주는 경우 많은 것으로 보인다.


