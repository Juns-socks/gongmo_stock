import os
import sys                        # system specific parameters and functions : 파이썬 스크립트 관리
from PyQt5.QtWidgets import *     # GUI의 그래픽적 요소를 제어       하단의 terminal 선택, activate py37_32,  pip install pyqt5,   전부다 y
from PyQt5 import uic             # ui 파일을 가져오기위한 함수
from PyQt5.QtCore import *        # eventloop/스레드를 사용 할 수 있는 함수 가져옴.

################# 직접 만든 파일들
from kiwoom import Kiwoom          # 키움증권 함수/공용 방 (싱글턴)
from account import Account      # 계좌평가잔고내역 가져오기
from randomforest import Randomforest            # 데이터 가져오기
from trade import Trade         # 자동매매 시작


form_class = uic.loadUiType("gui.ui")[0]             # 만들어 놓은 ui 불러오기

class Login_Machnine(QMainWindow, QWidget, form_class):

    def __init__(self, *args, **kwargs):

        print("Login Machine 실행합니다.")
        super(Login_Machnine, self).__init__(*args, **kwargs)
        form_class.__init__(self)
        self.setUI()                                         # UI 초기값 셋업 반드시 필요

        ### 초기 셋팅
        self.label_11.setText(str("총매입금액"))
        self.label_12.setText(str("총평가금액"))
        self.label_13.setText(str("추정예탁자산"))
        self.label_14.setText(str("총평가손익금액"))
        self.label_15.setText(str("총수익률(%)"))

        #### 기타 함수
        self.login_event_loop = QEventLoop()  # 이때 QEventLoop()는 block 기능을 가지고 있다.

        ####키움증권 로그인 하기
        self.k = Kiwoom()
        self.set_signal_slot()
        self.signal_login_commConnect()

        #####이벤트 생성 및 진행
        self.call_account.clicked.connect(self.c_acc)          #계좌정보가져오기
        self.R_F.clicked.connect(self.G_data)             #데이터 가져오기
        self.Auto_Start.clicked.connect(self.auto)            #자동매매 시작

    def setUI(self):
        self.setupUi(self)                # UI 초기값 셋업

    def set_signal_slot(self):
        self.k.kiwoom.OnEventConnect.connect(self.login_slot)

    def signal_login_commConnect(self):
        self.k.kiwoom.dynamicCall("CommConnect()")
        self.login_event_loop.exec_()  # 로그인이 완료될 때까지 계속 반복됨

    def login_slot(self, errCode):
        if errCode == 0:
            print("로그인 성공")
            self.statusbar.showMessage("로그인 성공")
            self.get_account_info()                    # 로그인시 계좌정보 가져오기

        elif errCode == 100:
            print("사용자 정보교환 실패")
        elif errCode == 101:
            print("서버접속 실패")
        elif errCode == 102:
            print("버전처리 실패")
        self.login_event_loop.exit()  # 로그인이 완료되면 로그인 창을 닫는다.

    def get_account_info(self):
        account_list = self.k.kiwoom.dynamicCall("GetLoginInfo(String)", "ACCNO")

        for n in account_list.split(';'):
            self.accComboBox.addItem(n)

    def c_acc(self):
        print("선택 계좌 정보 가져오기")
        h1 = Account(self)
        h1.start()

    def G_data(self):
        print("random forest")
        h2 = Randomforest(self)
        h2.start()


    def auto(self):
        print("자동매매 시작")
        h3 = Trade(self)
        h3.start()

if __name__=='__main__':
    app = QApplication(sys.argv)
    CH = Login_Machnine()
    CH.show()
    app.exec_()                      # 이벤트 루프