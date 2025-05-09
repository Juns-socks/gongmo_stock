import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import KernelPCA  # PCA 라이브러리 추가
from PyQt5.QtCore import *         # 쓰레드 함수를 불러온다.
from kiwoom import Kiwoom          # 로그인을 위한 클래스
from PyQt5.QtWidgets import *      #PyQt import


class Randomforest(QThread):
    def __init__(self, parent):   # 부모의 윈도우 창을 가져올 수 있다.
        super().__init__(parent)
        self.parent = parent

        self.k = Kiwoom()

        #빈칸으로 입력하면 안됨
        종목코드 = self.parent.searchItemTextEdit2.toPlainText()
        종목명 = self.k.kiwoom.dynamicCall("GetMasterCodeName(QString)", 종목코드)
        기관경쟁률 = float(self.parent.com_rate.toPlainText())
        시가총액 = float(self.parent.com_price.toPlainText())
        유통비율 = float(self.parent.move_rate.toPlainText())
        의무보유확약비율 = float(self.parent.must_rate.toPlainText())
        매수수량 = int(self.parent.buy_quantity.toPlainText())
        매수가격 = int(self.parent.buy_price.toPlainText())
        유통물량 = 시가총액 * 유통비율 / 100

        # 엑셀 파일 경로 설정
        file_path = "공모주데이터E1.xlsx"  # 파일 경로 변경
        # 엑셀 파일에서 데이터 로드
        T = pd.read_excel(file_path, sheet_name='Sheet1')

        def app1(x):
            if x["시가수익률"] > 0.3:
                return 1
            else:
                return 0

        def app2(x):
            if x["고가수익률"] > 0.15:
                return 1
            else:
                return 0

        T["유통물량"] = T["시가총액"] * T["유통비율"] / 100

        def app3(x):
            if x["스팩주"] == "ㅇ":
                if x["유통물량"] < 80:
                    return 1
                else:
                    return 0
            elif x["유통물량"] < 300:
                return 1
            else:
                return 0


        T["수익계수"] = T.apply(app1, axis=1)
        T["시초가매수"] = T.apply(app2, axis=1)
        T["유통계수"] = T.apply(app3, axis=1)

        T["미국지수"] = (T["Nasdaq_등락률"] + T['S&P_등락률']) / 2
        T['로그시가총액'] = np.log(T['시가총액'])

        # T = T[T['스팩주'].isna() | (T['스팩주'] != 'ㅇ')]
        # T = T[T['이전상장'].isna() | (T['이전상장'] != 'ㅇ')]

        # 결측값 처리
        # T['로그매출액'].fillna(T['로그매출액'].mean(), inplace=True)
        T['로그유통물량'] = np.log(T['유통물량'])

        pca_features = T[['유통물량', '시가총액', '공모가']]
        scaler = StandardScaler()
        pca_features_scaled = scaler.fit_transform(pca_features)

        # Kernel PCA 적용 (비선형 차원 축소)
        kpca = KernelPCA(n_components=1, kernel='rbf', gamma=0.1)  # rbf 커널 사용, gamma는 커널 파라미터
        principal_component = kpca.fit_transform(pca_features_scaled)

        # 첫 번째 주성분을 A로 저장
        T['A'] = principal_component

        결과 = '시초가매수'
        넣는거 = ['의무보유확약', '유통계수', '기관경쟁률']
        총넣는거 = ['의무보유확약', '유통계수', '기관경쟁률', 결과]

        data = T[총넣는거].dropna()
        # 독립 변수(X)와 종속 변수(y) 정의
        X = data[넣는거]
        y = data[결과]  # y를 1차원 배열로 변환
        # 인덱스 저장
        indices = y.index

        # 64 -> 0.65 ,
        random_state_index = 64
        # 학습 데이터와 테스트 데이터 분리
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2,
                                                                                         random_state=random_state_index)

        # 데이터 스케일링 (필요한 경우)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 랜덤 포레스트 모델 생성 (이진 분류로 가정)
        model = RandomForestClassifier(n_estimators=150, random_state=random_state_index)
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

        def predict_new_data(new_data):
            try:
                # 입력 데이터 정리
                input_df = pd.DataFrame([new_data])
                input_scaled = scaler.transform(input_df)  # 스케일링

                # 예측 수행
                prediction = model.predict(input_scaled)
                predicted_class = prediction[0]

                # 예측 결과 출력
                print(f"예측 결과 (수익계수): {predicted_class}")
                return predicted_class
            except Exception as e:
                print(f"Error in prediction: {e}")

        if "스팩" in 종목명:
            if 유통물량 < 80:
                유통계수 = 1
            else:
                유통계수 = 0
        else:
            if 유통물량 < 300:
                유통계수 = 1
            else:
                유통계수 = 0

        new_data_example = {
            '기관경쟁률': 기관경쟁률,
            '의무보유확약': 의무보유확약비율,
            '유통계수': 유통계수
        }

        # 새로운 데이터 예측
        self.k.go_buy = predict_new_data(new_data_example)

        if self.k.go_buy == 1:
            self.k.buy_list.update({종목코드: {}})
            self.k.sell_list.update({종목코드: {}})
            self.k.buy_list[종목코드].update({'매수수량': 매수수량})
            self.k.buy_list[종목코드].update({'매수가격': 매수가격})
            self.buy_price = self.k.buy_list[종목코드]['매수가격']
            print(종목명, "가 매수리스트에 들어갔습니다.")

        else:
            print(종목명, "가 매수리스트에 안들어갔습니다.")