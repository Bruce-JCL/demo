# -*- coding: utf-8 -*-

import cv2
import threading
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys

class Display(QWidget):
    def __init__(self):
        super(Display, self).__init__()

        self.setWindowTitle("视频录制")
        self.resize(1000, 800)

        self.open_btn = QPushButton("开始录制")
        self.close_btn = QPushButton("结束录制")
        layout_btn = QHBoxLayout()
        layout_btn.addWidget(self.open_btn)
        layout_btn.addWidget(self.close_btn)

        self.display_label = QLabel()
        #self.display_label.setGeometry(QRect())

        layout = QVBoxLayout()
        layout.addWidget(self.display_label)
        layout.addLayout(layout_btn)


        self.setLayout(layout)


        self.open_btn.clicked.connect(self.start_record)
        self.close_btn.clicked.connect(self.stop_record)

        
        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def start_record(self):
        # rtsp格式为"rtsp://用户名:密码@ip地址"
        self.cap = cv2.VideoCapture("rtsp://admin:aidlux123@192.168.110.186:554/h264/ch1/main/av_stream")

        # 创建视频显示线程
        th = threading.Thread(target=self.video_record)
        th.start()
    
    def stop_record(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def video_record(self):
        self.open_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(frame.shape[1],frame.shape[0],frame.shape[2])
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.display_label.setPixmap(QPixmap.fromImage(img))
            cv2.waitKey(10)

            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.display_label.clear()
                self.close_btn.setEnabled(False)
                self.open_btn.setEnabled(True)
                break

    # 添加中文的确认退出提示框1
    def closeEvent(self, event):
        # 创建一个消息盒子（提示框）
        quitMsgBox = QMessageBox()
        quitMsgBox.move(300, 300)
        # 设置提示框的标题
        quitMsgBox.setWindowTitle('确认提示')
        # 设置提示框的内容
        quitMsgBox.setText('你确认退出吗？')
        # 设置按钮标准，一个yes一个no
        quitMsgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # 获取两个按钮并且修改显示文本
        buttonY = quitMsgBox.button(QMessageBox.Yes)
        buttonY.setText('确定')
        buttonN = quitMsgBox.button(QMessageBox.No)
        buttonN.setText('取消')
        quitMsgBox.exec_()
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if quitMsgBox.clickedButton() == buttonY:
            # 关闭事件设为触发，关闭视频播放
            self.stopEvent.set()

            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    display = Display()
    display.show()
    sys.exit(app.exec_())

