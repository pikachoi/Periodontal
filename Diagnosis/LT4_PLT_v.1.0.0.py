# -*- coding: utf-8 -*-
import sys
import math
import os
import cv2
import json
import glob
import pandas as pd
import numpy as np
from ultralytics import YOLO
from shapely.geometry import LineString
from itertools import chain
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from functools import partial
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings(action='ignore')

form_class = uic.loadUiType("LT4_PLT_v.1.0.0.ui")[0]

class MouseTracker(QtCore.QObject):
    positionChanged = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, o, e):
        if o is self.widget and e.type() == QtCore.QEvent.MouseMove:
            self.positionChanged.emit(e.pos())
        return super().eventFilter(o, e)

class QtGUI(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.size_x = 1024
        self.size_y = 512
        globals()['df_line'] = pd.DataFrame(index=range(0), columns=['시작 좌표', '종료 좌표', 'pbl 교점', 'cej 교점', 'cej 유형', '비율', '길이'])
        globals()['df_pbl'] = pd.DataFrame(index=range(0), columns=['좌표'])
        globals()['df_cej_up'] = pd.DataFrame(index=range(0), columns=['좌표'])
        globals()['df_cej_low'] = pd.DataFrame(index=range(0), columns=['좌표'])
        self.model_tooth = YOLO("./best_class.pt")
        self.model_pbl = YOLO("./best_pbl_240122.pt")
        self.model_cej = YOLO("./best_cejl_240122.pt")
        self.initUI()

    def initUI(self):
        tracker = MouseTracker(self.label_img)
        tracker.positionChanged.connect(self.on_positionChanged)

        self.table_file.setColumnCount(1)
        self.table_file.setHorizontalHeaderItem(0, QTableWidgetItem('파일 명'))
        self.table_file.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_file.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_file.itemDoubleClicked.connect(self.tableFileDbClicked)
        header = self.table_file.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)

        for table in list(filter(lambda x: 'df' in x, globals())):
            eval(f'self.table_{table.split("df_")[-1]}.itemDoubleClicked.connect(partial(self.tableDataDbClicked, (\'{table.split("df_")[-1]}\')))')
            eval(f'self.table_{table.split("df_")[-1]}.setColumnCount(len(globals()["df_{table.split("df_")[-1]}"].columns))')
            eval(f'self.table_{table.split("df_")[-1]}.setHorizontalHeaderLabels(list(globals()["df_{table.split("df_")[-1]}"].columns))')
            eval(f'self.table_{table.split("df_")[-1]}.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)')
            eval(f'self.table_{table.split("df_")[-1]}.verticalHeader().setVisible(False)')
            eval(f'self.table_{table.split("df_")[-1]}.setShowGrid(False)')
            for i in range(len(eval(f'globals()["df_{table.split("df_")[-1]}"].columns'))):
                eval(f'self.table_{table.split("df_")[-1]}.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)')

        self.btn_add.clicked.connect(self.addList)
        self.btn_pbl_download.clicked.connect(partial(self.downloadFile,'pbl'))
        self.btn_cej_download.clicked.connect(partial(self.downloadFile,'cej'))
        self.btn_total_download.clicked.connect(partial(self.downloadFile,'total'))
        self.btn_line.clicked.connect(self.lineDetection)
        self.btn_calculation.clicked.connect(self.numericalCalculation)
        self.btn_grade.clicked.connect(self.grading)
        self.btn_test.clicked.connect(self.testgs)
        self.btn_test.setCheckable(True)

        self.txt_age.textChanged.connect(self.txtChanged)

        self.rbtn_line_draw.setChecked(True)

    def txtChanged(self):
        self.txt_score.setText('')
        self.txt_grade.setText('')

    def grading(self):
        if len(list(filter(lambda x: pd.isna(x) == False, globals()['df_line']['비율']))) != 0 and self.txt_age.toPlainText().isdigit():
            self.drawImg('total')
            self.numericalCalculation()
            df_grade = globals()['df_line'][pd.isna(globals()['df_line']['비율']) == False]
            df_grade['score'] = round(df_grade['비율'] / int(self.txt_age.toPlainText()), 3)
            df_grade['grade'] = None
            for i, row in df_grade.iterrows():
                if row['score'] <= 0.5:
                    df_grade['grade'][i] = 'A'
                    color_A = (255, 228, 0)
                    h, s, v = mcolors.rgb_to_hsv(color_A)
                    new_s = row['score'] + 0.5
                    new_v = 1

                    color_grade = mcolors.hsv_to_rgb((h, new_s, new_v))
                    color_grade = tuple(map(lambda x: x * 255, color_grade))
                elif row['score'] <= 1:
                    df_grade['grade'][i] = 'B'
                    color_B = (0, 255, 0)
                    h, s, v = mcolors.rgb_to_hsv(color_B)
                    new_s = row['score']
                    new_v = 1

                    color_grade = mcolors.hsv_to_rgb((h, new_s, new_v))
                    color_grade = tuple(map(lambda x: x * 255, color_grade))
                else:
                    df_grade['grade'][i] = 'C'
                    if row['score'] >= 1.5:
                        color_grade = (255, 0, 0)
                    else:
                        color_C = (255, 0, 0)
                        h, s, v = mcolors.rgb_to_hsv(color_C)
                        new_s = row['score'] - 0.5
                        new_v = 1

                        color_grade = mcolors.hsv_to_rgb((h, new_s, new_v))
                        color_grade = tuple(map(lambda x: x * 255, color_grade))

                if row['cej 유형'] == 'up':
                    cv2.putText(self.current_img, df_grade['grade'][i], (row['시작 좌표'][0] - 5, row['시작 좌표'][1] - 5),2, 0.4, color_grade, 1, cv2.LINE_AA)
                    cv2.putText(self.current_img, f"{str(int(row['비율']))}%", (int((row['시작 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['시작 좌표'][1] + row['pbl 교점'][1]) / 2)),
                                3, 0.3, color_grade, 1, cv2.LINE_AA)
                else:
                    cv2.putText(self.current_img, df_grade['grade'][i], (row['종료 좌표'][0] - 5, row['종료 좌표'][1] + 15),2, 0.4, color_grade, 1, cv2.LINE_AA)
                    cv2.putText(self.current_img, f"{str(int(row['비율']))}%", (int((row['종료 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['종료 좌표'][1] + row['pbl 교점'][1]) / 2)),
                                3, 0.3, color_grade, 1, cv2.LINE_AA)

            height, width, bytesPerComponent = self.current_img.shape
            bytesPerLine = 3 * width
            QImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            pixmap = pixmap.scaled(self.size_x, self.size_y)
            self.label_img.setPixmap(pixmap)
            self.label_img.setCursor(Qt.CrossCursor)

            result = max(list(df_grade['score']))

            self.txt_score.setText(str(result))
            self.txt_grade.setText(df_grade[df_grade['score'] == result]['grade'].values[0])

    def testgs(self):
        try:
            if self.btn_test.isChecked():
                self.gs_image = cv2.resize(self.img, (self.size_x, self.size_y))

                pbl_img = np.zeros((512, 1024), dtype=np.uint8) + 255
                pbl_seg = list(globals()['df_pbl']['좌표'])
                cv2.fillPoly(pbl_img, [np.array(pbl_seg, np.int32)], 0)

                cej_img = np.zeros((512, 1024), dtype=np.uint8)
                cej_up_seg = list(globals()['df_cej_up']['좌표'])
                cv2.fillPoly(cej_img, [np.array(cej_up_seg, np.int32)], 255)

                cej_down_seg = list(globals()['df_cej_low']['좌표'])
                cv2.fillPoly(cej_img, [np.array(cej_down_seg, np.int32)], 255)

                tooth_img = np.zeros((512, 1024), dtype=np.uint8)
                for v in self.results[0]:
                    tooth_seg = list(v.masks.xy)[0].tolist()
                    cv2.fillPoly(tooth_img, [np.array(tooth_seg, np.int32)], 255)

                final_img = tooth_img - cej_img - pbl_img

                _, binary_image = cv2.threshold(final_img, 128, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    if 200 <= cv2.contourArea(contour) <= 10000:
                        cv2.fillPoly(self.gs_image, [np.array(contour, np.int32)], 255)

                alpha = 0.7
                overlay = cv2.resize(self.img, (self.size_x, self.size_y))
                cv2.addWeighted(overlay, alpha, self.gs_image, 1 - alpha, 0, self.gs_image)

                height, width, bytesPerComponent = self.gs_image.shape
                bytesPerLine = 3 * width
                QImg = QImage(self.gs_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg)
                pixmap = pixmap.scaled(self.size_x, self.size_y)
                self.label_img.setPixmap(pixmap)
                self.label_img.setCursor(Qt.CrossCursor)
            else:
                height, width, bytesPerComponent = self.current_img.shape
                bytesPerLine = 3 * width
                QImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg)
                pixmap = pixmap.scaled(self.size_x, self.size_y)
                self.label_img.setPixmap(pixmap)
                self.label_img.setCursor(Qt.CrossCursor)
        except:
            self.btn_test.setChecked(False)


    def drawAxis(self, box, slope, intercept):
        [box_x, box_y, box_w, box_h] = box

        x1 = (box_y - intercept) / slope
        x2 = ((box_y + box_h) - intercept) / slope

        intersection_points = []

        if box_x <= x1 <= (box_x + box_w):
            intersection_points.append((int(x1), box_y))
        if box_x <= x2 <= (box_x + box_w):
            intersection_points.append((int(x2), box_y + box_h))

        y1 = slope * box_x + intercept
        y2 = slope * (box_x + box_w) + intercept
        if box_y <= y1 <= (box_y + box_h):
            intersection_points.append((box_x, int(y1)))
        if box_y <= y2 <= (box_y + box_h):
            intersection_points.append((box_x + box_w, int(y2)))

        return intersection_points

    def resize_and_pad(self, width, height, image):
        max_size = max(width, height)
        padded_image = np.full((max_size, max_size, 3), 0, dtype=np.uint8)
        x_offset = (max_size - width) // 2
        y_offset = (max_size - height) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
        return padded_image

    def lineDetection(self):
        try:
            self.btn_test.setChecked(False)
            if len(globals()['df_line']) == 0 and len(self.current_img) != 0:
                grade_list = ['A', 'B', 'C']
                datagen = ImageDataGenerator(rescale=1. / 255)
                detect_img = cv2.resize(self.img, (self.size_x, self.size_y))
                cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB, detect_img)
                self.results = self.model_tooth.predict(detect_img, conf=0.6)
                for v in self.results[0]:
                    box_x, box_y, box_w, box_h = int(v.boxes.data[0][0]), int(v.boxes.data[0][1]), (
                                int(v.boxes.data[0][2]) - int(v.boxes.data[0][0])), (
                                int(v.boxes.data[0][3]) - int(v.boxes.data[0][1]))
                    segmentation_points = np.array(list(v.masks.xy))
                    mask = np.zeros((512, 1024), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(segmentation_points, np.int32)], 255)
                    moments = cv2.moments(mask)
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
                    x1 = cx + np.cos(angle)
                    y1 = cy + np.sin(angle)
                    x2 = cx - np.cos(angle)
                    y2 = cy - np.sin(angle)
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    axis_line = self.drawAxis([box_x, box_y, box_w, box_h], slope, intercept)
                    globals()['df_line'].loc[len(globals()['df_line'])] = {'시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}

                self.append_table('line')
                self.drawImg('total')
        except:
            pass

    def distance(self, x0, y0, x1, y1):
        sq1 = (x0 - x1) * (x0 - x1)
        sq2 = (y0 - y1) * (y0 - y1)
        distance = round(math.sqrt(sq1 + sq2), 3)
        return distance

    def numericalCalculation(self):
        try:
            self.btn_test.setChecked(False)
            self.drawImg('total')
            if all(list(map(lambda x : len(x) > 0, [globals()['df_line'],globals()['df_pbl'],globals()['df_cej_up'],globals()['df_cej_low']]))):
                polyline_pbl = LineString(globals()['df_pbl']['좌표'].tolist()+ [globals()['df_pbl']['좌표'].tolist()[0]])
                polyline_cej_up = LineString(globals()['df_cej_up']['좌표'].tolist() + [globals()['df_cej_up']['좌표'].tolist()[0]])
                polyline_cej_low = LineString(globals()['df_cej_low']['좌표'].tolist() + [globals()['df_cej_low']['좌표'].tolist()[0]])
                for i, row in globals()['df_line'].iterrows():
                    line = LineString([row['시작 좌표'], row['종료 좌표']])
                    intersection_pbl = line.intersection(polyline_pbl)
                    intersection_cej_up = line.intersection(polyline_cej_up)
                    intersection_cej_low = line.intersection(polyline_cej_low)
                    if intersection_pbl.geom_type == 'Point' and intersection_cej_up.geom_type != intersection_cej_low.geom_type:
                        globals()['df_line']['cej 교점'] = globals()['df_line']['cej 교점'].astype(object)
                        globals()['df_line']['pbl 교점'] = globals()['df_line']['pbl 교점'].astype(object)
                        cv2.circle(self.current_img, [int(intersection_pbl.x), int(intersection_pbl.y)], 2, (255, 255, 255),-1)
                        globals()['df_line']['pbl 교점'][i] = [int(intersection_pbl.x), int(intersection_pbl.y)]
                        if intersection_cej_up.geom_type == 'Point' and intersection_cej_low.geom_type == 'LineString':
                            kp = intersection_cej_up
                            direction = 'up'
                        elif intersection_cej_up.geom_type == 'Point' and intersection_cej_low.geom_type == 'MultiPoint':
                            kp = max(intersection_cej_low.geoms, key=lambda point: point.y)
                            direction = 'low'
                        elif intersection_cej_low.geom_type == 'Point' and intersection_cej_up.geom_type == 'LineString':
                            kp = intersection_cej_low
                            direction = 'low'
                        elif intersection_cej_low.geom_type == 'Point' and intersection_cej_up.geom_type == 'MultiPoint':
                            kp = min(intersection_cej_up.geoms, key=lambda point: point.y)
                            direction = 'up'
                        elif intersection_cej_up.geom_type == 'MultiPoint' and intersection_cej_low.geom_type == 'LineString':
                            kp = min(intersection_cej_up.geoms, key=lambda point: point.y)
                            direction = 'up'
                        elif intersection_cej_low.geom_type == 'MultiPoint' and intersection_cej_up.geom_type == 'LineString':
                            kp = max(intersection_cej_low.geoms, key=lambda point: point.y)
                            direction = 'low'

                        cv2.circle(self.current_img, [int(kp.x), int(kp.y)], 2,(255, 255, 255), -1)
                        globals()['df_line']['cej 교점'][i] = [int(kp.x), int(kp.y)]

                        if direction == 'up':
                            start_kp = min(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                            cv2.circle(self.current_img, start_kp, 2, (92, 209, 229), -1)
                            globals()['df_line']['cej 유형'][i] = 'up'
                        else:
                            start_kp = max(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                            cv2.circle(self.current_img, start_kp, 2, (165, 102, 255), -1)
                            globals()['df_line']['cej 유형'][i] = 'low'

                        loss_distance = self.distance(int(intersection_pbl.x), int(intersection_pbl.y), int(kp.x), int(kp.y))
                        globals()['df_line']['길이'][i] = loss_distance
                        globals()['df_line']['비율'][i] = int(loss_distance / self.distance(start_kp[0], start_kp[1], int(kp.x), int(kp.y)) * 100)

                self.append_table('line')

                max_ratio = globals()['df_line'][globals()['df_line']['비율'] == max(globals()['df_line']['비율'])]
                cv2.line(self.current_img, max_ratio['pbl 교점'].tolist()[0], max_ratio['cej 교점'].tolist()[0], (170, 0, 255), 4, lineType=cv2.LINE_AA)
                cv2.circle(self.current_img, max_ratio['pbl 교점'].tolist()[0], 2,(255, 255, 0), -1)
                cv2.circle(self.current_img, max_ratio['cej 교점'].tolist()[0], 2, (255, 255, 0), -1)

                height, width, bytesPerComponent = self.current_img.shape
                bytesPerLine = 3 * width
                QImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg)
                pixmap = pixmap.scaled(self.size_x, self.size_y)
                self.label_img.setPixmap(pixmap)
                self.label_img.setCursor(Qt.CrossCursor)
        except:
            pass

    def downloadFile(self, typ):
        if typ == 'total':
            download_list = ['pbl', 'cej_up', 'cej_low']
        else:
            download_list = list(filter(lambda x : typ in x, ['pbl', 'cej_up', 'cej_low']))

        if len(list(filter(lambda x : eval(f'len(df_{x})') == 0, download_list))) == 0:
            os.makedirs(f'{os.path.dirname(self.file_name)}/labeling', exist_ok=True)
            gray_img = np.zeros((512, 1024), dtype=np.uint8)
            for v in download_list:
                if typ == 'total':
                    cv2.polylines(gray_img, [np.array(globals()[f'df_{v}']['좌표'].tolist())], True, 255, 1,lineType=cv2.LINE_AA)
                else:
                    cv2.fillPoly(gray_img, [np.array(globals()[f'df_{v}']['좌표'].tolist())], color=255)

            result, encoded_total = cv2.imencode('.png', gray_img)
            with open(f'{os.path.dirname(self.file_name)}/labeling/{os.path.basename(self.file_name).split(".")[0]}_{typ}.jpg', mode='w+b') as f:
                encoded_total.tofile(f)

            txt_list = ['pbl', 'cej_up', 'cej_low']
            txt_final = ''
            for v in download_list:
                txt_final += f"{txt_list.index(v)} {' '.join(list(chain.from_iterable(list(map(lambda x: [str(x[0] / self.size_x), str(x[-1] / self.size_y)],globals()[f'df_{v}']['좌표'].tolist())))))}\n"
            with open(f'{os.path.dirname(self.file_name)}/labeling/{os.path.basename(self.file_name).split(".")[0]}_{typ}.txt', 'w') as file:
                file.write(txt_final)

    def on_positionChanged(self, pos):
        self.label_x = pos.x()
        self.label_y = pos.y()
        standard = 50

        try:
            if self.btn_test.isChecked():
                self.magnifier_img = self.gs_image
            else:
                self.magnifier_img = self.current_img

            if len(self.magnifier_img) != 0:
                if self.label_x < standard and self.label_y < standard:
                    image_crop = self.magnifier_img[(0):(standard * 2), (0):(standard * 2)].copy()
                    cv2.circle(image_crop, (self.label_x, self.label_y), 1, (0, 255, 0), 1)
                elif self.label_x > self.size_x - standard and self.label_y > self.size_y - standard:
                    image_crop = self.magnifier_img[(self.size_y - standard * 2):(self.size_y), (self.size_x - standard * 2):(self.size_x)].copy()
                    cv2.circle(image_crop, (self.label_x - (self.size_x - standard * 2), self.label_y - (self.size_y - standard * 2)), 1, (0, 255, 0), 1)
                elif self.label_x < standard and self.label_y > self.size_y - standard:
                    image_crop = self.magnifier_img[(self.size_y - standard * 2):(self.size_y), (0):(standard * 2)].copy()
                    cv2.circle(image_crop, (self.label_x, self.label_y - (self.size_y - standard * 2)), 1, (0, 255, 0), 1)
                elif self.label_x > self.size_x - standard and self.label_y < standard:
                    image_crop = self.magnifier_img[(0):(standard * 2), (self.size_x - standard * 2):(self.size_x)].copy()
                    cv2.circle(image_crop, (self.label_x - (self.size_x - standard * 2), self.label_y), 1, (0, 255, 0), 1)
                elif self.label_x < standard:
                    image_crop = self.magnifier_img[(self.label_y - standard):(self.label_y + standard), (0):(standard * 2)].copy()
                    cv2.circle(image_crop, (self.label_x, standard), 1, (0, 255, 0), 1)
                elif self.label_y < standard:
                    image_crop = self.magnifier_img[(0):(standard * 2), (self.label_x - standard):(self.label_x + standard)].copy()
                    cv2.circle(image_crop, (standard, self.label_y), 1, (0, 255, 0), 1)
                elif self.label_x > self.size_x - standard:
                    image_crop = self.magnifier_img[(self.label_y - standard):(self.label_y + standard), (self.size_x - standard * 2):(self.size_x)].copy()
                    cv2.circle(image_crop, (self.label_x - (self.size_x - standard * 2), standard), 1, (0, 255, 0), 1)
                elif self.label_y > self.size_y - standard:
                    image_crop = self.magnifier_img[(self.size_y - standard * 2):(self.size_y), (self.label_x - standard):(self.label_x + standard)].copy()
                    cv2.circle(image_crop, (standard, self.label_y - (self.size_y - standard * 2)), 1, (0, 255, 0), 1)
                else:
                    image_crop = self.magnifier_img[(self.label_y - standard):(self.label_y + standard), (self.label_x - standard):(self.label_x + standard)].copy()
                    cv2.circle(image_crop, (standard, standard), 1, (0, 255, 0), 1)

                height, width, bytesPerComponent = image_crop.shape
                bytesPerLine = 3 * width
                QImg = QImage(image_crop.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg)
                pixmap = pixmap.scaled(image_crop.shape[1] * 2, image_crop.shape[0] * 2)
                self.label_large.setPixmap(pixmap)
                self.label_large.setCursor(Qt.CrossCursor)
        except:
            pass

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            if any([self.rbtn_line_delete.isChecked(), self.rbtn_pbl_delete.isChecked(), self.rbtn_cej_up_delete.isChecked(),self.rbtn_cej_low_delete.isChecked()]):
                check_delete_btn = list(filter(lambda x: x.isChecked(),[self.rbtn_line_delete, self.rbtn_pbl_delete, self.rbtn_cej_up_delete,self.rbtn_cej_low_delete]))[0].objectName()
                delete_class = check_delete_btn.split('rbtn_')[-1].split('_delete')[0]
                try:
                    self.DeleteData(delete_class)
                except:
                    pass

    def resetUi(self):
        self.results = ''
        self.btn_test.setChecked(False)
        self.file_name = ''
        self.current_img = ''
        self.label_img.clear()
        self.label_large.clear()
        self.txt_age.setText('')
        self.txt_grade.setText('')
        self.txt_score.setText('')
        self.tab_grade.setCurrentIndex(0)
        for v in ['pbl', 'cej_up', 'cej_low']:
            globals()[f'{v}_row'] = ''
        globals()['df_line'] = pd.DataFrame(index=range(0), columns=['시작 좌표', '종료 좌표', 'pbl 교점', 'cej 교점', 'cej 유형', '비율', '길이'])
        globals()['df_pbl'] = pd.DataFrame(index=range(0), columns=['좌표'])
        globals()['df_cej_up'] = pd.DataFrame(index=range(0), columns=['좌표'])
        globals()['df_cej_low'] = pd.DataFrame(index=range(0), columns=['좌표'])
        self.append_table('total')

    def addList(self):
        self.resetUi()
        self.table_file.setRowCount(0)

        self.files = ''
        self.files = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./", QFileDialog.ShowDirsOnly)

        if self.files == '':
            return

        self.file_list = [file for file in os.listdir(self.files) if file.endswith('.png') or file.endswith('.jpg')]
        self.table_file.setRowCount(len(self.file_list))

        for i in range(len(self.file_list)):
            self.table_file.setItem(i, 0, QTableWidgetItem(self.file_list[i]))

    def tableFileDbClicked(self, e):
        self.resetUi()
        self.file_name = self.files + '/' + self.file_list[self.table_file.currentIndex().row()]

        img_array = np.fromfile(self.file_name, np.uint8)
        self.img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        self.img_height, self.img_width, _ = self.img.shape
        center_x, center_y = self.img_width // 2, self.img_height // 2
        cutting_ratio = 0.75
        self.left = int(center_x - cutting_ratio / 2 * self.img_width)
        self.right = int(center_x + cutting_ratio / 2 * self.img_width)
        self.top = int(center_y - cutting_ratio / 2 * self.img_height)
        self.bottom = int(center_y + cutting_ratio / 2 * self.img_height)
        self.img = self.img[self.top:self.bottom, self.left:self.right]
        self.current_img = cv2.resize(self.img, (self.size_x, self.size_y))
        cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB, self.current_img)
        try:
            self.results_pbl = self.model_pbl.predict(self.current_img)
            self.results_cej = self.model_cej.predict(self.current_img)

            globals()['df_pbl']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_pbl[0].masks.xy[0]))))

            if len(self.results_cej[0].boxes.cls.tolist()) == 2:
                if self.results_cej[0].boxes.cls.tolist()[0] == 1:
                    globals()['df_cej_up']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_cej[0].masks.xy[1]))))
                    globals()['df_cej_low']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_cej[0].masks.xy[0]))))
                else:
                    globals()['df_cej_up']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_cej[0].masks.xy[0]))))
                    globals()['df_cej_low']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_cej[0].masks.xy[1]))))
            elif len(self.results_cej[0].boxes.cls.tolist()) == 1:
                if self.results_cej[0].boxes.cls.tolist()[0] == 1:
                    globals()['df_cej_low']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_cej[0].masks.xy[0]))))
                else:
                    globals()['df_cej_up']['좌표'] = self.optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(self.results_cej[0].masks.xy[0]))))
            self.append_table('total')
            self.drawImg('total')
        except:
            pass

        height, width, bytesPerComponent = self.current_img.shape
        bytesPerLine = 3 * width
        QImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        pixmap = pixmap.scaled(self.size_x, self.size_y)
        self.label_img.setPixmap(pixmap)
        self.label_img.setCursor(Qt.CrossCursor)

    def optimizationSeg(self, segmentation_points):
        for v in segmentation_points:
            for point in list(filter(lambda x: abs(x[0] - v[0]) <= 10 and abs(x[1] - v[1]) <= 10, list(filter(lambda x: x != v, segmentation_points)))):
                segmentation_points.remove(point)
        return segmentation_points

    def tableDataDbClicked(self, typ):
        self.btn_test.setChecked(False)
        for v in ['line', 'pbl', 'cej_up', 'cej_low']:
            globals()[f'{v}_row'] = ''
        try:
            if eval(f'self.rbtn_{typ}_delete.isChecked()') == True:
                globals()[f'{typ}_row'] = eval(f'self.table_{typ}.currentIndex().row()')

                self.drawImg('total')

                if typ == 'line':
                    for id, v in enumerate(globals()['df_line']["시작 좌표"]):
                        if id == globals()['line_row']:
                            cv2.line(self.current_img, globals()['df_line']["시작 좌표"][id], globals()['df_line']["종료 좌표"][id],(255, 255, 0), 1, lineType=cv2.LINE_AA)
                else:
                    for i, row in globals()[f'df_{typ}'].iterrows():
                        if i == globals()[f'{typ}_row']:
                            cv2.circle(self.current_img, row['좌표'], 1, (255, 255, 0), -1)

            height, width, bytesPerComponent = self.current_img.shape
            bytesPerLine = 3 * width
            QImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            pixmap = pixmap.scaled(self.size_x, self.size_y)
            self.label_img.setPixmap(pixmap)
            self.label_img.setCursor(Qt.CrossCursor)
        except:
            pass

    def DeleteData(self, typ):
        self.btn_test.setChecked(False)
        if globals()[f'{typ}_row'] != '' and eval(f'self.rbtn_{typ}_delete.isChecked()') == True:
            try:
                globals()[f'df_{typ}'] = globals()[f'df_{typ}'].drop(index=globals()[f'{typ}_row'], axis=0)
                globals()[f'df_{typ}'] = globals()[f'df_{typ}'].reset_index(drop=True)
                self.append_table(f'{typ}')

                self.drawImg('total')

                globals()[f'{typ}_row'] = ''
            except:
                pass

    def drawImg(self, typ):
        self.btn_test.setChecked(False)
        self.current_img = cv2.resize(self.img, (self.size_x, self.size_y))
        cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB, self.current_img)

        if typ == 'total':
            table_list = ['pbl', 'cej_up', 'cej_low', 'line']
        else:
            table_list = [typ]

        for v in table_list:
            if v == 'line':
                for i, row in globals()['df_line'].iterrows():
                    cv2.line(self.current_img, row["시작 좌표"], row["종료 좌표"],(71, 200, 62), 1, lineType=cv2.LINE_AA)
                    cv2.circle(self.current_img, row["시작 좌표"], 1, (0, 0, 0), -1)
                    cv2.circle(self.current_img, row["종료 좌표"], 1, (0, 0, 0), -1)
            else:
                if v == 'pbl':
                    line_color = (255, 0, 0)
                    circle_color = (0, 0, 255)
                else:
                    line_color = (0, 0, 255)
                    circle_color = (255, 0, 0)
                cv2.polylines(self.current_img, [np.array(globals()[f'df_{v}']['좌표'].tolist())], True, line_color, 1,lineType=cv2.LINE_AA)
                for i, row in globals()[f'df_{v}'].iterrows():
                    cv2.circle(self.current_img, row['좌표'], 1, circle_color, -1)

        height, width, bytesPerComponent = self.current_img.shape
        bytesPerLine = 3 * width
        QImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        pixmap = pixmap.scaled(self.size_x, self.size_y)
        self.label_img.setPixmap(pixmap)
        self.label_img.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        try:
            self.x0 = ''
            self.y0 = ''
            self.x1 = ''
            self.y1 = ''
            for v in ['pbl', 'cej_up', 'cej_low']:
                globals()[f'{v}_modify_index'] = ''

            if len(self.current_img) != 0 and event.x() >= 20 and event.y() >= 20 and event.x() <= 1044 and event.y() <= 532:
                self.x0 = self.label_x
                self.y0 = self.label_y
                if any([self.rbtn_line_draw.isChecked(),self.rbtn_pbl_draw.isChecked(),self.rbtn_cej_up_draw.isChecked(),self.rbtn_cej_low_draw.isChecked()]):
                    check_draw_btn = list(filter(lambda x : x.isChecked(), [self.rbtn_line_draw,self.rbtn_pbl_draw,self.rbtn_cej_up_draw,self.rbtn_cej_low_draw]))[0].objectName()
                    draw_class = check_draw_btn.split('rbtn_')[-1].split('_draw')[0]

                    if draw_class != 'line':
                        if len(globals()[f'df_{draw_class}']) == 0:
                            globals()[f'df_{draw_class}'].loc[0] = {'좌표': [self.x0, self.y0]}

                    globals()[f'current_len_{draw_class}'] = len(globals()[f'df_{draw_class}'])
                elif any([self.rbtn_pbl_modify.isChecked(),self.rbtn_cej_up_modify.isChecked(),self.rbtn_cej_low_modify.isChecked()]):
                    check_modify_btn = list(filter(lambda x: x.isChecked(),[self.rbtn_pbl_modify, self.rbtn_cej_up_modify, self.rbtn_cej_low_modify]))[0].objectName()
                    modify_class = check_modify_btn.split('rbtn_')[-1].split('_modify')[0]
                    globals()[f'{modify_class}_modify_index'] = list(globals()[f'df_{modify_class}']['좌표']).index(list(filter(lambda x : abs(self.label_x - x[0]) <= 3 and abs(self.label_y - x[1]) <= 3, list(globals()[f'df_{modify_class}']['좌표'])))[0])
        except:
            pass

    def mouseMoveEvent(self, event):
        try:
            if self.label_x < 0:
                self.x1 = 0
            elif self.label_x >= self.size_x:
                self.x1 = self.size_x
            else:
                self.x1 = self.label_x

            if self.label_y < 0:
                self.y1 = 0
            elif self.label_y >= self.size_y:
                self.y1 = self.size_y
            else:
                self.y1 = self.label_y

            if self.x0 != '' and self.y0 != '':
                if any([self.rbtn_line_draw.isChecked(), self.rbtn_pbl_draw.isChecked(),self.rbtn_cej_up_draw.isChecked(), self.rbtn_cej_low_draw.isChecked()]):
                    check_draw_btn = list(filter(lambda x: x.isChecked(),[self.rbtn_line_draw, self.rbtn_pbl_draw, self.rbtn_cej_up_draw,self.rbtn_cej_low_draw]))[0].objectName()
                    draw_class = check_draw_btn.split('rbtn_')[-1].split('_draw')[0]

                    if draw_class != 'line':
                        globals()[f'df_{draw_class}'].loc[globals()[f'current_len_{draw_class}']] = {'좌표': [self.x1, self.y1]}
                    else:
                        globals()['df_line'].loc[globals()['current_len_line']] = {'시작 좌표': [self.x0, self.y0],'종료 좌표': [self.x1, self.y1]}

                    self.append_table(draw_class)
                    self.drawImg('total')

                elif any([self.rbtn_pbl_modify.isChecked(), self.rbtn_cej_up_modify.isChecked(), self.rbtn_cej_low_modify.isChecked()]):
                    check_modify_btn = list(filter(lambda x: x.isChecked(),[self.rbtn_pbl_modify, self.rbtn_cej_up_modify, self.rbtn_cej_low_modify]))[0].objectName()
                    modify_class = check_modify_btn.split('rbtn_')[-1].split('_modify')[0]

                    if globals()[f'{modify_class}_modify_index'] != '':
                        globals()[f'df_{modify_class}']['좌표'][globals()[f'{modify_class}_modify_index']] = [self.x1, self.y1]
                        self.append_table(modify_class)

                    self.drawImg('total')
        except:
            pass

    def append_table(self, typ):
        if typ == 'total':
            table_list = ['line', 'pbl', 'cej_up', 'cej_low']
        else:
            table_list = [typ]

        for v in table_list:
            eval(f'self.table_{v}.setColumnCount(len(globals()["df_{v}"].columns))')
            eval(f'self.table_{v}.setRowCount(len(globals()["df_{v}"].index))')
            for i in range(len(eval(f'globals()["df_{v}"].index'))):
                for j in range(len(eval(f'globals()["df_{v}"].columns'))):
                    eval(f'self.table_{v}.setItem(i, j, QTableWidgetItem(str(globals()["df_{v}"].iloc[i, j])))')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = QtGUI()
    ex.show()
    app.exec_()