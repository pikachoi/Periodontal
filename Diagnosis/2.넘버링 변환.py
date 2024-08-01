import pandas as pd
import cv2
from ultralytics import YOLO
import numpy as np
import math


model_tooth = YOLO("media/AI_model/best_class.pt")
model_quardrant = YOLO("media/AI_model/quardrant_RAdam.pt")
model_dentex = YOLO("media/AI_model/dentex_0425.pt")

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


def lineDetection(img):
    try:
        # df_dentex라는 데이터프레임을 id, segmentation, center_point 열로 생성
        df_dentex = pd.DataFrame(columns=['id', 'segmentation', 'center_point'])
        df_line = pd.DataFrame(index=range(0), columns=['id', '시작 좌표', '종료 좌표', 'pbl 교점', 'cej 교점', 'cej 유형', '비율', '길이'])
        class_list = ['molar', 'bridge', 'canine', 'implant', 'incisor', 'premolar']

        size_x = 1024
        size_y = 512

        img = cv2.imread(img)
        # 이미지를 사이즈에 맞게 리사이즈하고 BGR에서 RGB로 변환
        detect_img = cv2.resize(img, (size_x, size_y))
        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB, detect_img)
        
        # 사분면(quadrant) 예측 모델을 사용하여 이미지에서 예측 수행
        results_quardrant = model_quardrant.predict(detect_img, conf=0.6)
        
        # 예측 결과를 하나씩 순회
        for result in results_quardrant[0]:
            # 영역 ID를 문자열로 변환
            area_id = str(int(result.boxes.cls.tolist()[0]))
            
            # 세그멘테이션 포인트를 리스트로 변환
            segmentation_points = list(map(lambda x: [int(x[0]), int(x[1])], list(result.masks.xy)[0].tolist()))
            
            # 마스크 이미지 생성
            mask = np.zeros((size_y, size_x), dtype=np.uint8)
            cv2.fillPoly(mask, np.array([segmentation_points], dtype=np.int32), 255)
            
            # 마스크 반전 이미지 생성
            mask_inv = cv2.bitwise_not(mask)
            
            # 패딩된 검정 이미지 생성
            padding_image = np.full((size_y, size_x, 3), (0, 0, 0), dtype=np.uint8)
            
            # 패딩된 이미지와 반전 마스크를 비트 연산하여 이미지 생성
            image_padded = cv2.bitwise_and(padding_image, padding_image, mask=mask_inv)
            


            # 원본 이미지와 마스크를 사용하여 패딩된 이미지에 추가
            globals()[f'image_padded_{area_id}'] = cv2.add(image_padded, cv2.bitwise_and(detect_img, detect_img, mask=mask))
            
            # dentex 예측 모델을 사용하여 패딩된 이미지에서 예측 수행
            results_dentex = model_dentex.predict(globals()[f'image_padded_{area_id}'], conf=0.6)


            
            
            # dentex 예측 결과를 하나씩 순회
            for result_dentex in results_dentex[0]:
                # 박스 좌표를 가져옴
                box_x, box_y, box_w, box_h = int(result_dentex.boxes.data[0][0]), int(result_dentex.boxes.data[0][1]), (
                        int(result_dentex.boxes.data[0][2]) - int(result_dentex.boxes.data[0][0])), (
                        int(result_dentex.boxes.data[0][3]) - int(result_dentex.boxes.data[0][1]))
                
                # 치아 ID를 문자열로 변환
                tooth_id = str(int(result_dentex.boxes.cls.tolist()[0]))
                
                # 세그멘테이션 포인트를 리스트로 변환
                segmentation_points = list(map(lambda x: [int(x[0]), int(x[1])], list(result_dentex.masks.xy)[0].tolist()))
                
                # 데이터프레임에 결과 추가
                df_dentex.loc[len(df_dentex)] = {'id': f'{area_id}_{tooth_id}', 'segmentation': segmentation_points, 'center_point': [int(box_x + box_w / 2), int(box_y + box_h / 2)]}
        
        # 치아 예측 모델을 사용하여 이미지에서 예측 수행
        results = model_tooth.predict(detect_img, conf=0.6)
        
        # 중심점 리스트 초기화
        center_list = []
        
        # 예측 결과를 하나씩 순회
        for v in results[0]:
            # 박스 좌표를 가져옴
            box_x, box_y, box_w, box_h = int(v.boxes.data[0][0]), int(v.boxes.data[0][1]), (
                        int(v.boxes.data[0][2]) - int(v.boxes.data[0][0])), (
                        int(v.boxes.data[0][3]) - int(v.boxes.data[0][1]))
            
            # 중심점을 리스트에 추가
            center_list.append([int(box_x + box_w / 2), int(box_y + box_h / 2)])
            
            # 세그멘테이션 포인트를 numpy 배열로 변환
            segmentation_points = np.array(list(v.masks.xy))
            
            # 마스크 이미지 생성
            mask = np.zeros((512, 1024), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(segmentation_points, np.int32)], 255)
            
            # 모멘트 계산
            moments = cv2.moments(mask)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # 각도 계산
            angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
            
            # 선의 시작점과 끝점 계산
            x1 = cx + np.cos(angle)
            y1 = cy + np.sin(angle)
            x2 = cx - np.cos(angle)
            y2 = cy - np.sin(angle)
            
            # 기울기와 절편 계산
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # 축을 그리는 함수 호출
            axis_line = draw_axis([box_x, box_y, box_w, box_h], slope, intercept)
            
            # df_line 데이터프레임에 결과 추가
            df_line.loc[len(df_line)] = {'id' : class_list[int(v.boxes.cls.tolist()[0])], '시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}
        
        # df_dentex의 각 행을 순회
        for i, row in df_dentex.iterrows():
            # 중심점과 각 중심점 간의 거리 계산
            dist_list = list(map(lambda x: distance(row['center_point'][0], row['center_point'][1], x[0], x[1]), center_list))
            
            # 가장 가까운 중심점의 id를 업데이트
            index_to_update = dist_list.index(min(dist_list))
            df_line.loc[index_to_update, 'id'] = row['id']
            

        # df_line.to_csv('df_line.csv', index=False) 


        # 이미지에 중심점과 ID를 시각화
        for i, row in df_line.iterrows():
            center = center_list[i]
            tooth_id = row['id']
            cv2.circle(detect_img, (center[0], center[1]), 5, (0, 255, 0), -1)
            cv2.putText(detect_img, tooth_id, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 이미지 일회성으로 표시
        cv2.imshow("Image with IDs", detect_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    # 결과 예시
    # class_list ?
    img = "media/455_1.jpg"
    lineDetection(img)




