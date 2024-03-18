from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from shapely.geometry import LineString
import math

model_tooth = YOLO("media/AI_model/best_class.pt")
model_pbl = YOLO("media/AI_model/best_pbl_240122.pt")
model_cej = YOLO("media/AI_model/best_cejl_240122.pt")  

globals()['df_line'] = pd.DataFrame(index=range(0), columns=['시작 좌표', '종료 좌표', 'pbl 교점', 'cej 교점', 'cej 유형', '비율', '길이'])
globals()['df_pbl'] = pd.DataFrame(index=range(0), columns=['좌표'])
globals()['df_cej_up'] = pd.DataFrame(index=range(0), columns=['좌표'])
globals()['df_cej_low'] = pd.DataFrame(index=range(0), columns=['좌표'])

size_x = 1024
size_y = 512

# 포인트 감소
def optimizationSeg(segmentation_points):
    for v in segmentation_points:
        for point in list(filter(lambda x: abs(x[0] - v[0]) <= 10 and abs(x[1] - v[1]) <= 10, list(filter(lambda x: x != v, segmentation_points)))):
            segmentation_points.remove(point)
    return segmentation_points


def drawAxis(box, slope, intercept):
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


def distance(x0, y0, x1, y1):
    sq1 = (x0 - x1) * (x0 - x1)
    sq2 = (y0 - y1) * (y0 - y1)
    distance = round(math.sqrt(sq1 + sq2), 3)
    return distance


def numericalCalculation(current_img):
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
                cv2.circle(current_img, [int(intersection_pbl.x), int(intersection_pbl.y)], 2, (255, 255, 255),-1)
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

                cv2.circle(current_img, [int(kp.x), int(kp.y)], 2,(255, 255, 255), -1)
                globals()['df_line']['cej 교점'][i] = [int(kp.x), int(kp.y)]

                if direction == 'up':
                    start_kp = min(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                    cv2.circle(current_img, start_kp, 2, (92, 209, 229), -1)
                    globals()['df_line']['cej 유형'][i] = 'up'
                else:
                    start_kp = max(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                    cv2.circle(current_img, start_kp, 2, (165, 102, 255), -1)
                    globals()['df_line']['cej 유형'][i] = 'low'

                loss_distance = distance(int(intersection_pbl.x), int(intersection_pbl.y), int(kp.x), int(kp.y))
                globals()['df_line']['길이'][i] = loss_distance
                globals()['df_line']['비율'][i] = int(loss_distance / distance(start_kp[0], start_kp[1], int(kp.x), int(kp.y)) * 100)


def grading(current_img, age):
    if len(list(filter(lambda x: pd.isna(x) == False, globals()['df_line']['비율']))) != 0 and isinstance(age, int):
        df_grade = globals()['df_line'][pd.isna(globals()['df_line']['비율']) == False]
        df_grade['score'] = round(df_grade['비율'] / int(age), 3)
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
                cv2.putText(current_img, df_grade['grade'][i], (row['시작 좌표'][0] - 5, row['시작 좌표'][1] - 5),2, 0.4, color_grade, 1, cv2.LINE_AA)
                cv2.putText(current_img, f"{str(int(row['비율']))}%", (int((row['시작 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['시작 좌표'][1] + row['pbl 교점'][1]) / 2)),
                            3, 0.3, color_grade, 1, cv2.LINE_AA)
            else:
                cv2.putText(current_img, df_grade['grade'][i], (row['종료 좌표'][0] - 5, row['종료 좌표'][1] + 15),2, 0.4, color_grade, 1, cv2.LINE_AA)
                cv2.putText(current_img, f"{str(int(row['비율']))}%", (int((row['종료 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['종료 좌표'][1] + row['pbl 교점'][1]) / 2)),
                            3, 0.3, color_grade, 1, cv2.LINE_AA)


def draw_line(img_path):
    # 이미지 전처리
    file_name = img_path
    img_array = np.fromfile(file_name, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape
    center_x, center_y = img_width // 2, img_height // 2
    cutting_ratio = 0.75
    left = int(center_x - cutting_ratio / 2 * img_width)
    right = int(center_x + cutting_ratio / 2 * img_width)
    top = int(center_y - cutting_ratio / 2 * img_height)
    bottom = int(center_y + cutting_ratio / 2 * img_height)
    img = img[top:bottom, left:right]
    current_img = cv2.resize(img, (size_x, size_y))

    # 예측 수행
    results = model_tooth.predict(current_img, conf=0.6)
    results_pbl = model_pbl.predict(current_img) 
    results_cej = model_cej.predict(current_img)

    # 예측 좌표 저장
    globals()['df_pbl']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_pbl[0].masks.xy[0]))))
    if len(results_cej[0].boxes.cls.tolist()) == 2:
        if results_cej[0].boxes.cls.tolist()[0] == 1:
            globals()['df_cej_up']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[1]))))
            globals()['df_cej_low']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
        else:
            globals()['df_cej_up']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
            globals()['df_cej_low']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[1]))))
    elif len(results_cej[0].boxes.cls.tolist()) == 1:
        if results_cej[0].boxes.cls.tolist()[0] == 1:
            globals()['df_cej_low']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
        else:
            globals()['df_cej_up']['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
    
    # 결과에서 상자 좌표 가져오기
    for v in results[0]:
        box_x, box_y, box_w, box_h = int(v.boxes.data[0][0]), int(v.boxes.data[0][1]), (
                    int(v.boxes.data[0][2]) - int(v.boxes.data[0][0])), (
                    int(v.boxes.data[0][3]) - int(v.boxes.data[0][1]))
        
        segmentation_points = np.array(list(v.masks.xy))
        mask = np.zeros((size_y, size_x), dtype=np.uint8)
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
        
        axis_line = drawAxis([box_x, box_y, box_w, box_h], slope, intercept)

        # 축 좌표 저장
        globals()['df_line'].loc[len(globals()['df_line'])] = {'시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}

    numericalCalculation(current_img)
    age = 40
    grading(current_img, age)

    cv2.imshow("Results", current_img)
    print(current_img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = "media/Raw/453_1.jpg"
    draw_line(img_path)








        # 치아 외곽선 그리기
        # cv2.polylines(current_img, [np.array(segmentation_points, np.int32)], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)


    # for i, row in globals()['df_line'].iterrows():
    #     cv2.line(current_img, row["시작 좌표"], row["종료 좌표"],(0, 255, 0), 1, lineType=cv2.LINE_AA)
    #     cv2.circle(current_img, row["시작 좌표"], 5, (0, 0, 0), -1)
    #     cv2.circle(current_img, row["종료 좌표"], 5, (255, 255, 255), -1)

    # cv2.polylines(current_img, [np.array(globals()['df_pbl']['좌표'].tolist())], True, (255, 0, 0), 1,lineType=cv2.LINE_AA)
    # cv2.polylines(current_img, [np.array(globals()['df_cej_up']['좌표'].tolist())], True, (0, 0, 255), 1,lineType=cv2.LINE_AA)
    # cv2.polylines(current_img, [np.array(globals()['df_cej_low']['좌표'].tolist())], True, (0, 255, 255), 1,lineType=cv2.LINE_AA)



    # print("치아 라인-------------------------------------------------")
    # print(globals()['df_line'])
    # print("PBL-------------------------------------------------")
    # print(globals()['df_pbl'])
    # print("CEJ_UP-------------------------------------------------")
    # print(globals()['df_cej_up'])
    # print("CEJ_LOW-------------------------------------------------")
    # print(globals()['df_cej_low'])


    # 웹 코드
#     pbl_seg = [np.array(list(globals()['df_pbl']['좌표']), np.int32)]
#     cej_seg_up = [np.array(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[1]))), np.int32)]
#     cej_seg_low = [np.array(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))), np.int32)]

# for seg in pbl_seg:
#         cv2.polylines(current_img, [seg], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
#     for seg in cej_seg_up:
#         cv2.polylines(current_img, [seg], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
#     for seg in cej_seg_low:
#         cv2.polylines(current_img, [seg], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)