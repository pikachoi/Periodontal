from django.shortcuts import render, redirect
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from shapely.geometry import LineString
import math
from django.http import JsonResponse
import base64
import copy

model_tooth = YOLO("media/AI_model/best_class.pt")
model_pbl = YOLO("media/AI_model/best_pbl_240122.pt")
model_cej = YOLO("media/AI_model/best_cejl_240122.pt")  



def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()

def png_encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)  # PNG 형식으로 인코딩
    base64_encoded_image = base64.b64encode(buffer).decode('utf-8')
    return base64_encoded_image

# 포인트 감소
def optimizationSeg(segmentation_points):
    for v in segmentation_points:
        for point in list(filter(lambda x: abs(x[0] - v[0]) <= 10 and abs(x[1] - v[1]) <= 10, list(filter(lambda x: x != v, segmentation_points)))):
            segmentation_points.remove(point)
    return segmentation_points


def draw_axis(box, slope, intercept):
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
                # cv2.circle(current_img, [int(intersection_pbl.x), int(intersection_pbl.y)], 2, (255, 255, 255),-1)
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

                # cv2.circle(current_img, [int(kp.x), int(kp.y)], 2,(255, 255, 255), -1)
                globals()['df_line']['cej 교점'][i] = [int(kp.x), int(kp.y)]

                if direction == 'up':
                    start_kp = min(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                    # cv2.circle(current_img, start_kp, 2, (92, 209, 229), -1)
                    globals()['df_line']['cej 유형'][i] = 'up'
                else:
                    start_kp = max(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                    # cv2.circle(current_img, start_kp, 2, (165, 102, 255), -1)
                    globals()['df_line']['cej 유형'][i] = 'low'

                loss_distance = distance(int(intersection_pbl.x), int(intersection_pbl.y), int(kp.x), int(kp.y))
                globals()['df_line']['길이'][i] = loss_distance
                globals()['df_line']['비율'][i] = int(loss_distance / distance(start_kp[0], start_kp[1], int(kp.x), int(kp.y)) * 100)

                # globals()['df_line']['비율'][i] = loss_distance / distance(start_kp[0], start_kp[1], int(kp.x), int(kp.y)) * 100 # 반올림 하지 않음
                # globals()['df_line']['비율'][i] = round(globals()['df_line']['비율'][i], 2) # 소수점 두자리까지


def grading(current_img, age):
    if len(list(filter(lambda x: pd.isna(x) == False, globals()['df_line']['비율']))) != 0 and isinstance(age, int):
        df_grade = globals()['df_line'][pd.isna(globals()['df_line']['비율']) == False].copy()
        df_grade['score'] = df_grade['비율'].astype(float) / age
        df_grade['grade'] = df_grade['score'].apply(lambda x: 'A' if x <= 0.5 else ('B' if x <= 1 else 'C'))
        
        for i, row in df_grade.iterrows():
            if row['grade'] == 'A':
                color_grade = (255, 228, 0)
            elif row['grade'] == 'B':
                color_grade = (0, 255, 0)
            else:
                color_grade = (255, 0, 0)
            
            color_grade = tuple(int(c) for c in color_grade)

            # if row['cej 유형'] == 'up':
            #     text_pos_grade = (row['시작 좌표'][0] - 5, row['시작 좌표'][1] - 5)
            #     text_pos_ratio = (int((row['시작 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['시작 좌표'][1] + row['pbl 교점'][1]) / 2))
            # else:
            #     text_pos_grade = (row['종료 좌표'][0] - 5, row['종료 좌표'][1] + 15)
            #     text_pos_ratio = (int((row['종료 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['종료 좌표'][1] + row['pbl 교점'][1]) / 2))
            
            # cv2.putText(current_img, f"{row['grade']}", text_pos_grade, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_grade, 1)
            # cv2.putText(current_img, f"{str(int(row['비율']))}%", text_pos_ratio, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_grade, 1)

    return df_grade


def apply_mask_operations(results):
    # 완전 투명한 이미지 크기 설정
    img_height, img_width = 512, 1024
    transparent_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)  # 4채널 RGBA, 모든 채널을 0으로 초기화

    # 각 영역의 마스크를 생성
    cej_up_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cej_low_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pbl_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    tooth_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 영역별 마스크 적용 로직
    cv2.fillPoly(cej_up_mask, [np.array(globals()['df_cej_up']['좌표'].tolist(), dtype=np.int32)], 255)
    cv2.fillPoly(cej_low_mask, [np.array(globals()['df_cej_low']['좌표'].tolist(), dtype=np.int32)], 255)
    cv2.fillPoly(pbl_mask, [np.array(globals()['df_pbl']['좌표'].tolist(), dtype=np.int32)], 255)

    # 치아 마스크 생성
    for tooth in results[0].masks.xy:
        cv2.fillPoly(tooth_mask, [np.array(tooth, dtype=np.int32)], 255)

    # CEJ 마스크 합성
    cej_mask = cv2.bitwise_or(cej_up_mask, cej_low_mask)

    # PBL에서 CEJ 마스크 제외
    pbl_without_cej = cv2.bitwise_and(pbl_mask, cv2.bitwise_not(cej_mask))

    # 'a' (PBL without CEJ)와 'b' (치아 마스크)의 겹치는 영역 계산
    intersection_mask = cv2.bitwise_and(pbl_without_cej, tooth_mask)

    # 결과 시각화
    # 투명한 배경 위에 교차 영역을 녹색으로 표시 (녹색의 알파 값은 255로 불투명하게)
    transparent_img[intersection_mask == 255] = [0, 0, 255, 50]

    return transparent_img


def diagnosis_home(request):
    if request.method == "POST":
        try:
            globals()['df_line'] = pd.DataFrame(index=range(0), columns=['시작 좌표', '종료 좌표', 'pbl 교점', 'cej 교점', 'cej 유형', '비율', '길이'])
            globals()['df_pbl'] = pd.DataFrame(index=range(0), columns=['좌표'])
            globals()['df_cej_up'] = pd.DataFrame(index=range(0), columns=['좌표'])
            globals()['df_cej_low'] = pd.DataFrame(index=range(0), columns=['좌표'])

            size_x = 1024
            size_y = 512

            age_input = request.POST.get('age', None)
            age = int(age_input)

            image_file = request.FILES["imgfile"] 
            image_data = image_file.read()

            img_array = np.frombuffer(image_data, np.uint8)
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
            current_img_copy = copy.deepcopy(current_img)

            results = model_tooth.predict(current_img, conf=0.6)
            results_pbl = model_pbl.predict(current_img) 
            results_cej = model_cej.predict(current_img)

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

            cv2.polylines(current_img, [np.array(globals()[f'df_pbl']['좌표'].tolist())], True, (0, 255, 0), 1,lineType=cv2.LINE_AA)
            cv2.polylines(current_img, [np.array(globals()[f'df_cej_up']['좌표'].tolist())], True, (0, 0, 255), 1,lineType=cv2.LINE_AA)
            cv2.polylines(current_img, [np.array(globals()[f'df_cej_low']['좌표'].tolist())], True, (255, 0, 0), 1,lineType=cv2.LINE_AA)


            # ---------------------------------------------------------------- 마스크 이미지 생성
            mask_image = apply_mask_operations(results)
            # 마스크 이미지를 Base64로 인코딩
            encoded_mask_image = png_encode_image_to_base64(mask_image)
            # ----------------------------------------------------------------

            additional_data = {
                'df_cej_up': globals()['df_cej_up'].to_json(orient="records"),
                'df_cej_low': globals()['df_cej_low'].to_json(orient="records"),
                'df_pbl': globals()['df_pbl'].to_json(orient="records"),
                'mask_image': encoded_mask_image,  # 마스크 이미지 추가
            }

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
                
                axis_line = draw_axis([box_x, box_y, box_w, box_h], slope, intercept)

                # 축 좌표 저장
                globals()['df_line'].loc[len(globals()['df_line'])] = {'시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}


            for i, row in globals()['df_line'].iterrows():
                        cv2.line(current_img, row["시작 좌표"], row["종료 좌표"],(255, 255, 0), 1, lineType=cv2.LINE_AA)
                        # cv2.circle(current_img, row["시작 좌표"], 1, (0, 0, 0), -1)
                        # cv2.circle(current_img, row["종료 좌표"], 1, (0, 0, 0), -1)
            
            numericalCalculation(current_img)
            df_grade = grading(current_img, age)

            # ---------------------------------------------------------------- 크롭 이미지 생성
            # 최대 골소실률 치아의 바운딩 박스 정보 추출
            max_ratio_index = df_grade['비율'].idxmax()
            max_ratio_box = results[0].boxes.data[max_ratio_index]
            expand_ratio = 0.2  # 바운딩 박스를 10% 확장
            box_x = int(max_ratio_box[0])
            box_y = int(max_ratio_box[1])
            box_w = int(max_ratio_box[2] - max_ratio_box[0])
            box_h = int(max_ratio_box[3] - max_ratio_box[1])
            # 확장 로직 적용
            expand_w = int(box_w * expand_ratio)
            expand_h = int(box_h * expand_ratio)
            # 바운딩 박스 확장 및 이미지 경계 처리
            new_x = max(box_x - expand_w, 0)
            new_y = max(box_y - expand_h, 0)
            new_w = box_w + 2 * expand_w
            new_h = box_h + 2 * expand_h
            # 이미지 크기 제한 (예: 이미지 경계 넘어가지 않게 처리)
            new_w = min(new_w, current_img_copy.shape[1] - new_x)
            new_h = min(new_h, current_img_copy.shape[0] - new_y)
            # 이미지 크롭
            cropped_img = current_img_copy[new_y:new_y + new_h, new_x:new_x + new_w]
            # 크롭된 이미지 인코딩 및 기타 처리
            encoded_cropped_image = encode_image_to_base64(cropped_img)
            additional_data['cropped_image'] = encoded_cropped_image
            # ----------------------------------------------------------------

            # cv2.imshow("Results", current_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            encoded_image = encode_image_to_base64(current_img_copy)
            # encoded_image = encode_image_to_base64(current_img)
            graded_results = []

            for i, row in df_grade.iterrows():
                graded_results.append({
                    'encoded_image': encoded_image,
                    'cej_type': row['cej 유형'],
                    'start_coordinate': row['시작 좌표'],
                    'end_coordinate': row['종료 좌표'],
                    'pbl_intersection': row['pbl 교점'],
                    'cej_intersection': row['cej 교점'],
                    'ratio': row['비율'],
                    'score': row['score'],
                    'grade': row['grade'],
                })

            return JsonResponse({
                'graded_data': graded_results,
                'additional_data': additional_data
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    else:
        return render(request, "Diagnosis_main.html")