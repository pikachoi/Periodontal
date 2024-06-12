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
from datetime import datetime
from dateutil.relativedelta import relativedelta


model_tooth = YOLO("media/AI_model/best_class.pt")
model_pbl = YOLO("media/AI_model/best_pbl_240122.pt")
model_cej = YOLO("media/AI_model/best_cejl_240122.pt")  
model_quardrant = YOLO("media/AI_model/quardrant_RAdam.pt")
model_dentex = YOLO("media/AI_model/dentex_0425.pt")
class_list = ['molar', 'bridge', 'canine', 'implant', 'incisor', 'premolar']  # 예시
size_x = 1024
size_y = 512


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()


def png_encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)  # PNG 형식으로 인코딩
    base64_encoded_image = base64.b64encode(buffer).decode('utf-8')
    return base64_encoded_image


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


def numericalCalculation(df_line, df_pbl, df_cej_up, df_cej_low):
    if all(list(map(lambda x : len(x) > 0, [df_line, df_pbl, df_cej_up, df_cej_low]))):
        polyline_pbl = LineString(df_pbl['좌표'].tolist()+ [df_pbl['좌표'].tolist()[0]])
        polyline_cej_up = LineString(df_cej_up['좌표'].tolist() + [df_cej_up['좌표'].tolist()[0]])
        polyline_cej_low = LineString(df_cej_low['좌표'].tolist() + [df_cej_low['좌표'].tolist()[0]])
        for i, row in df_line.iterrows():
            line = LineString([row['시작 좌표'], row['종료 좌표']])
            intersection_pbl = line.intersection(polyline_pbl)
            intersection_cej_up = line.intersection(polyline_cej_up)
            intersection_cej_low = line.intersection(polyline_cej_low)
            if intersection_pbl.geom_type == 'Point' and intersection_cej_up.geom_type != intersection_cej_low.geom_type:
                df_line['cej 교점'] = df_line['cej 교점'].astype(object)
                df_line['pbl 교점'] = df_line['pbl 교점'].astype(object)
                df_line.at[i, 'pbl 교점'] = [int(intersection_pbl.x), int(intersection_pbl.y)]
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

                df_line.at[i, 'cej 교점'] = [int(kp.x), int(kp.y)]

                if direction == 'up':
                    start_kp = min(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                    df_line.at[i, 'cej 유형'] = 'up'
                else:
                    start_kp = max(row[['시작 좌표', '종료 좌표']], key=lambda point: point[-1])
                    df_line.at[i, 'cej 유형'] = 'low'

                loss_distance = distance(int(intersection_pbl.x), int(intersection_pbl.y), int(kp.x), int(kp.y))
                df_line.at[i, '길이'] = loss_distance
                df_line.at[i, '비율'] = int(loss_distance / distance(start_kp[0], start_kp[1], int(kp.x), int(kp.y)) * 100)

                # df_line.at[i, '비율'] = loss_distance / distance(start_kp[0], start_kp[1], int(kp.x), int(kp.y)) * 100 # 반올림 하지 않음
                # df_line.at[i, '비율'] = round(df_line.at[i, '비율'], 2) # 소수점 두자리까지

    return df_line


def grading(img, age, df_line):
    if len(list(filter(lambda x: pd.isna(x) == False, df_line['비율']))) != 0 and isinstance(age, int):
        df_grade = df_line[pd.isna(df_line['비율']) == False].copy()
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

            if row['cej 유형'] == 'up':
                text_pos_grade = (row['시작 좌표'][0] - 5, row['시작 좌표'][1] - 5)
                text_pos_ratio = (int((row['시작 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['시작 좌표'][1] + row['pbl 교점'][1]) / 2))
            else:
                text_pos_grade = (row['종료 좌표'][0] - 5, row['종료 좌표'][1] + 15)
                text_pos_ratio = (int((row['종료 좌표'][0] + row['pbl 교점'][0]) / 2) - 15, int((row['종료 좌표'][1] + row['pbl 교점'][1]) / 2))
            
            cv2.putText(img, f"{row['grade']}", text_pos_grade, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_grade, 1)  # 시각화
            cv2.putText(img, f"{str(int(row['비율']))}%", text_pos_ratio, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_grade, 1)  # 시각화

    return df_grade


def apply_mask_operations(results, df_cej_up, df_cej_low, df_pbl):
    # 완전 투명한 이미지 크기 설정
    img_height, img_width = 512, 1024
    transparent_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    # 각 영역의 마스크를 생성
    cej_up_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cej_low_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pbl_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    tooth_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 영역별 마스크 적용 로직
    cv2.fillPoly(cej_up_mask, [np.array(df_cej_up['좌표'].tolist(), dtype=np.int32)], 255)
    cv2.fillPoly(cej_low_mask, [np.array(df_cej_low['좌표'].tolist(), dtype=np.int32)], 255)
    cv2.fillPoly(pbl_mask, [np.array(df_pbl['좌표'].tolist(), dtype=np.int32)], 255)

    # 치아 마스크 생성
    for tooth in results[0].masks.xy:
        cv2.fillPoly(tooth_mask, [np.array(tooth, dtype=np.int32)], 255)

    # CEJ 마스크 합성
    cej_mask = cv2.bitwise_or(cej_up_mask, cej_low_mask)

    # PBL에서 CEJ 마스크 제외
    pbl_without_cej = cv2.bitwise_and(pbl_mask, cv2.bitwise_not(cej_mask))

    # PBL without CEJ 치아 마스크의 겹치는 영역 계산
    intersection_mask = cv2.bitwise_and(pbl_without_cej, tooth_mask)

    # 결과 시각화
    transparent_img[intersection_mask == 255] = [0, 0, 255, 50]

    return transparent_img


# single, time_series 공통
def common_fungtion(img_file, mode):
    df_line = pd.DataFrame(index=range(0), columns=['id', '시작 좌표', '종료 좌표', 'pbl 교점', 'cej 교점', 'cej 유형', '비율', '길이', 'ratio_difference', 'time_grade'])
    df_pbl = pd.DataFrame(index=range(0), columns=['좌표'])
    df_cej_up = pd.DataFrame(index=range(0), columns=['좌표'])
    df_cej_low = pd.DataFrame(index=range(0), columns=['좌표'])
    
    img = img_file.read()
    img_array = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape
    center_x, center_y = img_width // 2, img_height // 2
    cutting_ratio = 0.75
    left = int(center_x - cutting_ratio / 2 * img_width)
    right = int(center_x + cutting_ratio / 2 * img_width)
    top = int(center_y - cutting_ratio / 2 * img_height)
    bottom = int(center_y + cutting_ratio / 2 * img_height)
    img = img[top:bottom, left:right]
    img = cv2.resize(img, (size_x, size_y))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    if mode is 'ime_series':
        df_dentex = pd.DataFrame(columns=['id', 'segmentation', 'center_point'])
        results_quardrant = model_quardrant.predict(img, conf=0.6)
    elif mode is 'single':
        df_dentex = 'none'
        results_quardrant = 'none'

    results = model_tooth.predict(img, conf=0.6)
    results_pbl = model_pbl.predict(img) 
    results_cej = model_cej.predict(img)

    df_pbl['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_pbl[0].masks.xy[0]))))
    if len(results_cej[0].boxes.cls.tolist()) == 2:
        if results_cej[0].boxes.cls.tolist()[0] == 1:
            df_cej_up['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[1]))))
            df_cej_low['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
        else:
            df_cej_up['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
            df_cej_low['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[1]))))
    elif len(results_cej[0].boxes.cls.tolist()) == 1:
        if results_cej[0].boxes.cls.tolist()[0] == 1:
            df_cej_low['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))
        else:
            df_cej_up['좌표'] = optimizationSeg(list(map(lambda x: [int(x[0]), int(x[1])], list(results_cej[0].masks.xy[0]))))

    cv2.polylines(img, [np.array(df_pbl['좌표'].tolist())], True, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.polylines(img, [np.array(df_cej_up['좌표'].tolist())], True, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.polylines(img, [np.array(df_cej_low['좌표'].tolist())], True, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    return df_line, df_pbl, df_cej_up, df_cej_low, img, results, results_quardrant, df_dentex


def diagnosis_single(request):
    if request.method == "POST":
        mode = 'single'
        try:
            img_file = request.FILES["imgfile"] 
            age = int(request.POST.get('age'))
            df_line, df_pbl, df_cej_up, df_cej_low, img, results, results_quardrant, df_dentex = common_fungtion(img_file, mode)
            img_copy = copy.deepcopy(img)

            mask_image = apply_mask_operations(results, df_cej_up, df_cej_low, df_pbl)
            encoded_mask_image = png_encode_image_to_base64(mask_image)

            additional_data = {
                'df_cej_up': df_cej_up.to_json(orient="records"),
                'df_cej_low': df_cej_low.to_json(orient="records"),
                'df_pbl': df_pbl.to_json(orient="records"),
                'mask_image': encoded_mask_image, 
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

                df_line.loc[len(df_line)] = {'시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}

            for i, row in df_line.iterrows():
                cv2.line(img, row["시작 좌표"], row["종료 좌표"],(255, 255, 0), 1, lineType=cv2.LINE_AA)

            numericalCalculation(df_line, df_pbl, df_cej_up, df_cej_low)
            df_grade = grading(img, age, df_line)

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
            new_w = min(new_w, img_copy.shape[1] - new_x)
            new_h = min(new_h, img_copy.shape[0] - new_y)
            # 이미지 크롭
            cropped_img = img_copy[new_y:new_y + new_h, new_x:new_x + new_w]
            # 크롭된 이미지 인코딩 및 기타 처리
            encoded_cropped_image = encode_image_to_base64(cropped_img)
            additional_data['cropped_image'] = encoded_cropped_image
            # ----------------------------------------------------------------

            encoded_image = encode_image_to_base64(img)
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
        return render(request, "Diagnosis_single.html")
    

def presume_loss(df_0, df_1, date_0, date_1):
    # 'id' 기준으로 정렬하고 인덱스를 재설정
    df_0 = df_0.sort_values(by=['id']).reset_index(drop=True)
    df_1 = df_0.sort_values(by=['id']).reset_index(drop=True)
    
    # 'bridge', 'implant'를 제외한 id들의 리스트 생성
    # list_start = tuple(filter(lambda x: x not in ['bridge', 'implant'], df_0['id']))
    # list_current = tuple(filter(lambda x: x not in ['bridge', 'implant'], df_1['id']))
    
# id 리스트의 중복이 없고, 현재 리스트의 모든 요소가 시작 리스트에 있는지 확인
# if len(set(list_start)) == len(list_start) and len(set(list_current)) == len(list_current) and len(set(list_current) - set(list_start)) == 0:
    # 시작 날짜와 현재 날짜 사이의 경과 일수를 계산
    elapsed_date = int((date_1 - date_0).days)

    # 'ratio_difference' 열을 0으로 초기화하고 'time_grade' 열을 빈 문자열로 초기화
    df_1['ratio_difference'] = 0
    df_1['time_grade'] = ''
    
    # 시작 데이터프레임의 각 행을 반복
    for _, row in df_0.iterrows():
        # 'class_list'에 없고 '비율' 값이 NaN이 아닌 id를 찾음
        if row['id'] not in class_list and pd.isna(row['비율']) == False:
            # 현재 데이터프레임에서 해당 id를 가진 행을 찾습니다.  ----------------------- 여기서 실제 비율값이 들어가고 있지 않은듯..
            df_map = df_1[df_1['id'] == row['id']]
            
            # 해당 id가 현재 데이터프레임에 있고 '비율' 값이 NaN이 아닌 경우
            if len(df_map) != 0 and pd.isna(df_map['비율'].values[0]) == False:
                # 비율 차이를 계산하고 연간 비율로 환산하여 5년 기준으로 환산
                ratio_difference = (df_map['비율'].values[0] - row['비율']) / elapsed_date * 365 * 5
                # 비율 차이에 따라 등급을 매김
                if ratio_difference < 3:
                    time_grade = 'A'
                elif 3 <= ratio_difference < 10:
                    time_grade = 'B'
                else:
                    time_grade = 'C'
                # 'ratio_difference'와 'time_grade' 값을 현재 데이터프레임에 업데이트
                df_1['ratio_difference'][df_map.index[0]] = ratio_difference

                df_1['time_grade'][df_map.index[0]] = time_grade
            # 해당 id가 현재 데이터프레임에 없는 경우
            elif len(df_map) == 0:
                # 해당 id를 새로운 행으로 추가하고 'time_grade'를 'C'로 설정
                df_1.loc[len(df_1)] = {'id': row['id'], 'ratio_difference': None, 'time_grade': 'C'}

    return df_1


# 결과를 원하는 형식으로 변환
def format_relativedelta(rd):
    return f"{rd.years}y, {rd.months}m, {rd.days}d"


def time_series_processing(img_file, age):
    mode = 'ime_series'
    df_line, df_pbl, df_cej_up, df_cej_low, img, results_t, results_quardrant, df_dentex = common_fungtion(img_file, mode)

    image_padded_dict = {} 
    for result in results_quardrant[0]:
        area_id = str(int(result.boxes.cls.tolist()[0])) 
        segmentation_points = list(map(lambda x: [int(x[0]), int(x[1])], list(result.masks.xy)[0].tolist())) 
        mask = np.zeros((size_y, size_x), dtype=np.uint8)  
        cv2.fillPoly(mask, np.array([segmentation_points], dtype=np.int32), 255)
        mask_inv = cv2.bitwise_not(mask)  
        padding_image = np.full((size_y, size_x, 3), (0, 0, 0), dtype=np.uint8)
        image_padded = cv2.bitwise_and(padding_image, padding_image, mask=mask_inv) 
        image_padded_dict[area_id] = cv2.add(image_padded, cv2.bitwise_and(img, img, mask=mask))
        results_dentex = model_dentex.predict(image_padded_dict[area_id], conf=0.6)

        for result_dentex in results_dentex[0]:
            # 박스 좌표를 가져옴
            box_x, box_y, box_w, box_h = int(result_dentex.boxes.data[0][0]), int(result_dentex.boxes.data[0][1]), (
                    int(result_dentex.boxes.data[0][2]) - int(result_dentex.boxes.data[0][0])), (
                    int(result_dentex.boxes.data[0][3]) - int(result_dentex.boxes.data[0][1]))
            
            tooth_id = str(int(result_dentex.boxes.cls.tolist()[0]))  # 치아 ID를 문자열로 변환

            segmentation_points = list(map(lambda x: [int(x[0]), int(x[1])], list(result_dentex.masks.xy)[0].tolist()))  # 세그멘테이션 포인트를 리스트로 변환
            df_dentex.loc[len(df_dentex)] = {'id': f'{area_id}_{tooth_id}', 'segmentation': segmentation_points, 'center_point': [int(box_x + box_w / 2), int(box_y + box_h / 2)]}  # 데이터프레임에 결과 추가

    center_list = []
    for v in results_t[0]:
        # 박스 좌표를 가져옴
        box_x, box_y, box_w, box_h = int(v.boxes.data[0][0]), int(v.boxes.data[0][1]), (
                    int(v.boxes.data[0][2]) - int(v.boxes.data[0][0])), (
                    int(v.boxes.data[0][3]) - int(v.boxes.data[0][1]))
        center_list.append([int(box_x + box_w / 2), int(box_y + box_h / 2)])  # 중심점을 리스트에 추가
        segmentation_points = np.array(list(v.masks.xy))   # 세그멘테이션 포인트를 numpy 배열로 변환
        mask = np.zeros((512, 1024), dtype=np.uint8)  # 마스크 이미지 생성
        cv2.fillPoly(mask, [np.array(segmentation_points, np.int32)], 255)

        # 모멘트 계산
        moments = cv2.moments(mask) 
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])  # 각도 계산

        # 선의 시작점과 끝점 계산
        x1 = cx + np.cos(angle)
        y1 = cy + np.sin(angle)
        x2 = cx - np.cos(angle)
        y2 = cy - np.sin(angle)

        # 기울기와 절편 계산
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        axis_line = draw_axis([box_x, box_y, box_w, box_h], slope, intercept)

        # df_line 데이터프레임에 결과 추가
        df_line.loc[len(df_line)] = {'id' : class_list[int(v.boxes.cls.tolist()[0])], '시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}

    for i, row in df_dentex.iterrows():
        dist_list = list(map(lambda x: distance(row['center_point'][0], row['center_point'][1], x[0], x[1]), center_list)) 
        
        # 가장 가까운 중심점의 id를 업데이트
        index_to_update = dist_list.index(min(dist_list))
        df_line.loc[index_to_update, 'id'] = row['id']

    # 이미지에 중심점과 ID를 시각화
    for i, row in df_line.iterrows():
        center = center_list[i]
        tooth_id = row['id']
        cv2.line(img, row["시작 좌표"], row["종료 좌표"],(255, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (center[0], center[1]), 5, (0, 255, 0), -1)
        cv2.putText(img, tooth_id, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    numericalCalculation(df_line, df_pbl, df_cej_up, df_cej_low)
    df_grade = grading(img, age, df_line)
    encoded_image = encode_image_to_base64(img)

    # 이미지 일회성으로 표시
    cv2.imshow("Image with IDs", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return encoded_image, df_grade, df_line


def diagnosis_time_series(request):
    if request.method == "POST":
        try:
            img0 = request.FILES.get('img0')
            img1 = request.FILES.get('img1')

            age0 = int(request.POST.get('age0'))
            age1 = int(request.POST.get('age1'))

            date0 = request.POST.get('date0')
            date1 = request.POST.get('date1')
            date2 = request.POST.get('date2')

            date0 = datetime.strptime(date0, '%Y-%m-%d')
            date1 = datetime.strptime(date1, '%Y-%m-%d')
            date2 = datetime.strptime(date2, '%Y-%m-%d')
            
            encoded_image0, df_grade0, df_0 = time_series_processing(img0, age0)
            df_0.to_csv('df_0.csv', index=False) 

            encoded_image1, df_grade1, df_1 = time_series_processing(img1, age1)
            df_1.to_csv('df_1.csv', index=False) 

            df_00 = pd.read_csv('df_00.csv')
            df_100 = presume_loss(df_00, df_1, date0, date1)
            df_100.to_csv('presume_df_2.csv', index=False)



            # graded_results0 = []
            # for i, row in df_grade0.iterrows():
            #     graded_results0.append({
            #         'encoded_image': encoded_image0,
            #         'cej_type': row['cej 유형'],
            #         'start_coordinate': row['시작 좌표'],
            #         'end_coordinate': row['종료 좌표'],
            #         'pbl_intersection': row['pbl 교점'],
            #         'cej_intersection': row['cej 교점'],
            #         'ratio': row['비율'],
            #         'score': row['score'],
            #         'grade': row['grade'],
            #     })

            # graded_results1 = []
            # for i, row in df_grade1.iterrows():
            #     graded_results1.append({
            #         'encoded_image': encoded_image1,
            #         'cej_type': row['cej 유형'],
            #         'start_coordinate': row['시작 좌표'],
            #         'end_coordinate': row['종료 좌표'],
            #         'pbl_intersection': row['pbl 교점'],
            #         'cej_intersection': row['cej 교점'],
            #         'ratio': row['비율'],
            #         'score': row['score'],
            #         'grade': row['grade'],
            #     })
            
            # # ------------------------------------------------------------------------------------------------------------------------------------------


            # dff_1_and_0 = date1 - date0
            # dff_2_and_0 = date2 - date0

            # # 날짜 차이 (일수, 이후 수식에 사용)
            # dff_1_and_0 = dff_1_and_0.days
            # dff_2_and_0 = dff_2_and_0.days

            # # 날짜 차이 (년 월 일)
            # ymd_dff_1_and_0 = relativedelta(date1, date0)
            # ymd_dff_2_and_0 = relativedelta(date2, date0)

            # ymd_dff_1_and_0 = format_relativedelta(ymd_dff_1_and_0)
            # ymd_dff_2_and_0 = format_relativedelta(ymd_dff_2_and_0)

            # print(ymd_dff_1_and_0)
            # print(ymd_dff_2_and_0)

            return JsonResponse({'ymd_dff_1_and_0': 'ymd_dff_1_and_0',
                                 'ymd_dff_2_and_0': 'ymd_dff_2_and_0',})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return render(request, "Diagnosis_time_series.html")


def clinic(request):
        return render(request, "clinic.html")

