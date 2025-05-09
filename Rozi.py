import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 엑셀 파일에서 데이터 로드
T = pd.read_excel("공모주데이터E1.xlsx",engine = 'openpyxl')


def app1(x):
    if (x["시가수익률"]) > 0.2:
        return 1
    else:
        return 0


T["수익계수"] = T.apply(app1, axis=1)

data = T[['로그기관경쟁률', '기관경쟁률', '의무보유확약', '보호예수비율', '유통비율', '로그보호예수비율', '로그의무보유확약',
          'Nasdaq_등락률', '공모가위치1', '공모가', '로그공모가', '보호예수비율역수', '수익계수']].dropna()

# 독립 변수(X)와 종속 변수(y) 정의
X = data[['로그기관경쟁률', '보호예수비율역수','의무보유확약']]
y = data[['수익계수']]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# SMOTE 적용
smote = SMOTE(random_state=16)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 데이터 스케일링 (필요한 경우)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# 로지스틱 회귀 모델 생성 (이진 분류로 가정)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# 예측 수행
y_pred = model.predict(X_test)

# 결과 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')
