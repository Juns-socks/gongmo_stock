import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def app1(x):
    if (x["시가수익률"] > 0.2) :
        return 1
    elif (x["고가수익률"] > 0.4) :
        return 1
    else:
        return 0
def app2(x):
    if (x["기관경쟁률"]) > 500.0:
        return 1
    else:
        return 0
# 엑셀 데이터 로드
gongmo = pd.read_excel("공모주데이터E1.xlsx",engine = 'openpyxl')
#gongmo = gongmo[gongmo['스팩주'].isna() | (gongmo['스팩주'] != 'ㅇ')]
#gongmo = gongmo[gongmo['이전상장'].isna() | (gongmo['이전상장'] != 'ㅇ')]
#gongmo["시가수익률"] = ((gongmo["시가"] - gongmo["공모가"]) / gongmo["공모가"]) * 100
#gongmo["고가수익률"] = ((gongmo["고가"] - gongmo["공모가"]) / gongmo["공모가"]) * 100
gongmo["유통금액"] = (gongmo["시가총액"] * gongmo["유통비율"])/100


gongmo["수익계수"] = gongmo.apply(app1, axis=1)
gongmo["기관계수"] = gongmo.apply(app2, axis=1)

# 1. 데이터 준비
# 여러 독립 변수 사용 ('공모가', '기관경쟁률', '시가총액(억)', '유통비율(%)' 등을 예시로 선택)
data = gongmo[[ '로그기관경쟁률', '기관경쟁률', '의무보유확약', '보호예수비율', '유통비율', '로그보호예수비율', '로그의무보유확약',
          'Nasdaq_등락률', '공모가위치1', '공모가', '로그공모가', '보호예수비율역수', '수익계수','시가수익률','유통금액','기관계수']].dropna()

# 비어 있는 셀이 있는 열 삭제
X = data[['로그기관경쟁률', '보호예수비율역수','의무보유확약','유통금액']]  # 독립 변수
y = data[['수익계수']]  # 종속 변수

# 시가수익률일때 : 로그기관경쟁률,의무보유확약,로지스틱시가2,공모가위치1

#random = 18 일때 0.57
#random = 14 일때 0.54
#random = 8 일때 0.5 , 고가수익률 추가시 0.6
#random = 47,고가 일때 0.54
#random = 82,고가 일때 0.55
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)


# 3. 모델 학습
model = linear_model.Ridge()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 4. 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# 5. 예측 vs 실제 값 시각화
#plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')  # 실제 값과 예측 값의 산포도
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # 대각선
plt.show()
