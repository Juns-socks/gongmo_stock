import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.decomposition import KernelPCA

# 엑셀 파일 경로 설정
file_path = "공모주데이터E1.xlsx"
save_path = "output\SVM분석결과.xlsx"

T = pd.read_excel(file_path, sheet_name='Sheet1')


def app1(x):
    if x["시가수익률"] > 0.3:
        return 1
    else:
        return 0


def app2(x):
    if (x["고가수익률"] > 0.40): return 1
    if (0.4 >= x['고가수익률'] > 1.0):
        return 2
    else:
        return 0


T["수익계수"] = T.apply(app1, axis=1)

T["미국지수"] = (T["Nasdaq_등락률"] + T['S&P_등락률']) / 2
T['로그시가총액'] = np.log(T['시가총액'])
T["유통물량"] = T["시가총액"] * T["유통비율"]

# T = T[T['스팩주'].isna() | (T['스팩주'] != 'ㅇ')]
# T = T[T['이전상장'].isna() | (T['이전상장'] != 'ㅇ')]

# 결측값 처리
T['로그매출액'].fillna(T['로그매출액'].mean(), inplace=True)
T['로그유통물량'] = np.log(T['유통물량'])

pca_features = T[['유통물량', '시가총액', '공모가']]
scaler = StandardScaler()
pca_features_scaled = scaler.fit_transform(pca_features)

# Kernel PCA 적용 (비선형 차원 축소)
kpca = KernelPCA(n_components=1, kernel='rbf', gamma=0.1)  # rbf 커널 사용, gamma는 커널 파라미터
principal_component = kpca.fit_transform(pca_features_scaled)

# 첫 번째 주성분을 A로 저장
T['A'] = principal_component

feature_candidates = [
    '로그기관경쟁률', '의무보유확약', '보호예수비율',
    'A', '주간사규모',
    '공모가위치', '미국지수']
# 필요한 데이터만 선택
data = T[['로그기관경쟁률', '기관경쟁률', '의무보유확약', '보호예수비율', '유통비율',
          '로그보호예수비율', '로그의무보유확약', 'Nasdaq_등락률', 'S&P_등락률', '공모가위치1',
          '공모가', '로그공모가', '보호예수비율역수', '미국지수', '수익계수', '공모가위치', 'A']].dropna()

X = data[['로그기관경쟁률', '의무보유확약', '보호예수비율', '공모가위치']]
y = data['수익계수']

# 인덱스 저장
indices = y.index

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.3,
                                                                                 random_state=42)

# SMOTE 적용
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM 모델 생성 및 학습 (이진 분류)
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 결과 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# 인덱스와 예측 결과를 데이터프레임으로 저장
results = pd.DataFrame({
    'Index': indices_test,
    'Actual': y_test.values.ravel(),
    'Predicted': y_pred
})

# 잘못 분류된 인덱스 추출
misclassified_0_indices = results[(results['Actual'] == 0) & (results['Predicted'] == 1)]['Index'].tolist()
misclassified_1_indices = results[(results['Actual'] == 1) & (results['Predicted'] == 0)]['Index'].tolist()
classified_0_indices = results[(results['Actual'] == 0) & (results['Predicted'] == 0)]['Index'].tolist()
classified_1_indices = results[(results['Actual'] == 1) & (results['Predicted'] == 1)]['Index'].tolist()

# 잘못 분류된 데이터 추출 (종목 이름 포함)
misclassified_0_data = T.loc[misclassified_0_indices]
misclassified_1_data = T.loc[misclassified_1_indices]
classified_0_data = T.loc[classified_0_indices]
classified_1_data = T.loc[classified_1_indices]

# 종목 이름 추출 및 출력
misclassified_0_names = misclassified_0_data['종목'].tolist()
misclassified_1_names = misclassified_1_data['종목'].tolist()
classified_0_names = classified_0_data['종목'].tolist()
classified_1_names = classified_1_data['종목'].tolist()

print(f'1인데 잘못 분류된 종목 이름: {misclassified_1_names}')
print(f'0인데 잘못 분류된 종목 이름: {misclassified_0_names}')

# 엑셀 파일로 저장
with pd.ExcelWriter(save_path) as writer:
    misclassified_1_data.to_excel(writer, sheet_name='1W', index=False)
    misclassified_0_data.to_excel(writer, sheet_name='0W', index=False)
    classified_1_data.to_excel(writer, sheet_name='1R', index=False)
    classified_0_data.to_excel(writer, sheet_name='0R', index=False)

print(f'SVM분석결과 {save_path}에 저장되었습니다.')
