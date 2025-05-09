from PyQt5.QtCore import *                  # 쓰레드 함수를 불러온다.
from kiwoom import Kiwoom                   # 로그인을 위한 클래스
from kiwoomType import *
from PyQt5.QtTest import *           # 시간관련 함수


class Trade(QThread):
    def __init__(self, parent):   # 부모의 윈도우 창을 가져올 수 있다.
        super().__init__(parent)
        self.parent = parent

        self.k = Kiwoom()

        account = self.parent.accComboBox.currentText()  # 콤보박스 안에서 가져오는 부분
        self.account_num = account

        self.realType = RealType()                # 실시간 FID 번호를 모아두는 곳
        self.buy_cnt = 0
        self.screen_num = 5000
        self.k.kiwoom.OnReceiveRealData.connect(self.realdata_slot)  # 실시간 데이터를 받아오는 곳

        # self.k.go_buy = 1
        # self.k.buy_list.update({"005930": {}})
        # self.k.buy_list["005930"].update({'매수수량': 1})
        # self.k.buy_list["005930"].update({'매수가격': 55000})
        # self.buy_price = self.k.buy_list["005930"]['매수가격']
        # self.k.sell_list.update({"005930": {}})
        # self.buy_cnt=1
        # self.k.buy_list["005930"].update({'현재가': 55000})
        # 매수 매도
        if self.k.go_buy == 1:
            self.buy()
            self.check()
            while True:
                if self.k.go_sell == 1:
                    break
                QTest.qWait(5000)

            self.sell()

    def buy(self):
        for sCode in self.k.buy_list.keys():
            print("매수 시작 %s" % sCode)
            self.buy_cnt = self.k.buy_list[sCode]['매수수량']
            order_success1 = self.k.kiwoom.dynamicCall(
                "SendOrder(QString, QString, QString ,int, QString, int, int, QString, QString)",
                ["신규매수", "2345", self.account_num, 1, sCode,
                 self.buy_cnt, self.k.buy_list[sCode]['매수가격'],
                 self.realType.SENDTYPE['거래구분']['지정가'], ""])

            if order_success1 == 0:
                print("주문 전달 성공")
            else:
                print("주문 전달 실패")

    def sell(self):
        for sCode in self.k.sell_list.keys():
            print("매도 시작 %s" % sCode)

            order_success2 = self.k.kiwoom.dynamicCall(
                "SendOrder(QString, QString, QString ,int, QString, int, int, QString, QString)",
                ["신규매도", "3456", self.account_num, 2, sCode,
                 self.buy_cnt, self.k.buy_list[sCode]["현재가"],
                 self.realType.SENDTYPE['거래구분']['지정가'], ""])

            if order_success2 == 0:
                print("주문 전달 성공")
            else:
                print("주문 전달 실패")

    def check(self):
        for code in self.k.buy_list.keys():
            self.k.kiwoom.dynamicCall("DisconnectRealData(QString)", self.screen_num)
            self.k.kiwoom.dynamicCall("SetRealReg(QString, QString, QString, QString)", self.screen_num, code, '10',
                                      "0")
            print("체크 시작합니다")
            #now_price = self.k.buy_list[code]['현재가']

    def realdata_slot(self, sCode, sRealType, sRealData):  # 실시간으로 서버에서 데이터들이 날라온다.
        if sRealType == "주식체결" and sCode in self.k.buy_list:
            fid5 = self.realType.REALTYPE[sRealType]['현재가']  # 매도쪽에 첫번재 부분(시장가)
            now_price = self.k.kiwoom.dynamicCall("GetCommRealData(QString, int)", sCode, fid5)
            now_price = abs(int(now_price))
            print("현재가 =", now_price)

            # 포트폴리오 종목코드마다 아래 실시간 데이터를 입력
            self.k.buy_list[sCode].update({'현재가': now_price})
            if now_price >= self.buy_price * 1.15 or now_price <= self.buy_price * 0.9:
                self.k.go_sell = 1
                print("매도시작")

            QTest.qWait(5000)
