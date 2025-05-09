import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 엑셀 파일 경로 설정
#file_path = r"C:\Users\seung\OneDrive\바탕 화면\공모주데이터E1.xlsx"

# 엑셀 파일에서 데이터 로드
T = pd.read_excel("공모주데이터E1.xlsx",engine = 'openpyxl')

def app1(x):
    if(x["시가수익률"] > 0.3): return 1
    else: return 0
def app2(x):
    if (x["고가수익률"] < 0):
        return 0
    elif (x["고가수익률"] < 0.1):
        return 1
    elif (x["고가수익률"] < 0.2):
        return 2
    elif (x["고가수익률"] < 0.3):
        return 3
    elif (x["고가수익률"] < 0.5):
        return 2
    elif (x["고가수익률"] < 0.7):
        return 2
    elif (x["고가수익률"] < 1.0):
        return 6
    elif (x["고가수익률"] < 1.4):
        return 6
    elif (x["고가수익률"] < 1.9):
        return 8
    elif (x["고가수익률"] < 2.5):
        return 9
    elif (x["고가수익률"] < 3.0):
        return 10
    else:
        return 11


# T1 = T[T['스팩주'].isna() | (T['스팩주'] != 'ㅇ')]
# T = T1[T1['이전상장'].isna() | (T1['이전상장'] != 'ㅇ')]
T["수익계수"] = T.apply(app1, axis=1)
#T["수익계수"] = T.apply(app2, axis=1)
# 필요한 데이터만 선택
data = T[['로그기관경쟁률','기관경쟁률', '의무보유확약', '보호예수비율', '유통비율',
          '로그보호예수비율', '로그의무보유확약', 'Nasdaq_등락률', '공모가위치1',
          '공모가', '로그공모가', '보호예수비율역수', '수익계수']].dropna()


# 독립 변수(X)와 종속 변수(y) 정의
X = data[['기관경쟁률', '보호예수비율역수', '로그공모가']]  # 사용할 변수 선택
y = data[['수익계수']]  # 종속 변수

# 데이터 분할
X_train_resampled, X_test, y_train_resampled, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# SMOTE를 사용하여 소수 클래스 증강


# 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# 딥러닝 모델 구성
model = Sequential()

# 입력 레이어 및 첫 번째 은닉 레이어
model.add(Dense(32, input_dim=X_train_resampled.shape[1], activation='relu'))
model.add(Dropout(0.3))

# 두 번째 은닉 레이어
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))

# 출력 레이어 (이진 분류의 경우 sigmoid 활성화 함수 사용)
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2)

# 테스트 데이터 예측
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)  # 0.5를 임계값으로 사용해 이진화

# 결과 평가
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 정확도 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# 손실 시각화
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
