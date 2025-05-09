import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def app1(x):
    if(x["시가수익률"] > 50): return 1
    else: return 0
def app2(x):
    if(x["고가수익률"] <0): return 0
    elif(x["고가수익률"]<0.2): return 1
    elif (x["고가수익률"] < 0.4): return 2
    elif (x["고가수익률"] < 0.6): return 3
    elif (x["고가수익률"] < 0.8): return 4
    elif (x["고가수익률"] < 1.0): return 5
    elif (x["고가수익률"] < 1.2): return 6
    elif (x["고가수익률"] < 1.4): return 7
    elif (x["고가수익률"] < 1.6): return 8
    elif (x["고가수익률"] < 1.8): return 9
    elif (x["고가수익률"] < 2.0): return 10
    elif (x["고가수익률"] < 2.2):return 11
    elif (x["고가수익률"] < 2.4): return 12
    elif (x["고가수익률"] < 2.6): return 13
    elif (x["고가수익률"] < 2.8): return 14
    else: return 15

gongmo = pd.read_excel("공모주.xlsx",engine = 'openpyxl')
#gongmo = gongmo[gongmo['스팩주'].isna() | (gongmo['스팩주'] != 'ㅇ')]
#gongmo = gongmo[gongmo['이전상장'].isna() | (gongmo['이전상장'] != 'ㅇ')]
gongmo["시가수익률"] = (gongmo["시가"]-gongmo["공모가"] / gongmo["공모가"]) * 100
gongmo["고가수익률"] = (gongmo["고가"]-gongmo["공모가"] / gongmo["공모가"]) * 100
gongmo["유통금액"] = (gongmo["시가총액"] * gongmo["유통비율"])/100

gongmo["수익계수"] = gongmo.apply(app1, axis=1)

gongmo["TEST"] = gongmo["의무보유확약"] + gongmo["기관경쟁률"]/10 + gongmo["보호예수비율"]

data = gongmo[['기관경쟁률', '시가수익률']].dropna()

# 독립 변수(X)와 종속 변수(y) 정의
X = data[['기관경쟁률']]
y = data['시가수익률']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 결과 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 회귀 계수와 절편 출력
print(f'회귀 계수: {model.coef_[0]}')
print(f'절편: {model.intercept_}')

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual',s=20)  # 실제 값
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')  # 예측 값
plt.xlabel('x')
plt.ylabel('siga')
plt.title('test')
plt.legend()
plt.show()
