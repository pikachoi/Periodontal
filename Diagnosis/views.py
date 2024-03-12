from django.shortcuts import render, redirect
from ultralytics import YOLO
import cv2
import numpy as np
from django.http import JsonResponse


model_tooth = YOLO("media/AI_model/best_class.pt")
model_pbl = YOLO("media/AI_model/best_pbl_240122.pt")
model_cej = YOLO("media/AI_model/best_cejl_240122.pt")  

size_x = 1024
size_y = 512


def draw_axis(box, slope, intercept):
    [box_x, box_y, box_w, box_h] = box

    # 선분과 상자의 교차점 찾기
    x1 = (box_y - intercept) / slope
    x2 = ((box_y + box_h) - intercept) / slope

    intersection_points = []

    # 교차점이 상자 안에 있는지 확인하고 저장
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


def diagnosis_home(request):
    if request.method == "POST":
        image_file = request.FILES["imgfile"] 
        image_data = image_file.read()

        nparr = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size_x, size_y)) 

        results = model_tooth.predict(img, conf=0.6)
        results_pbl = model_pbl.predict(img) 
        results_cej = model_cej.predict(img)

        pbl_seg = np.array(list(map(lambda x: [int(x[0]), int(x[1])], results_pbl[0].masks.xy[0])), np.int32)
        cej_seg_up = np.array(list(map(lambda x: [int(x[0]), int(x[1])], results_cej[0].masks.xy[1])), np.int32)
        cej_seg_low = np.array(list(map(lambda x: [int(x[0]), int(x[1])], results_cej[0].masks.xy[0])), np.int32)

        tooth_data = []

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

            tooth_dict = {
                "segmentation_points": segmentation_points.tolist(),
                "axis_line": axis_line,
            }
            tooth_data.append(tooth_dict)
            
            cv2.line(img, axis_line[0], axis_line[1], (50, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(img, [np.array(segmentation_points, np.int32)], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pbl_seg], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(img, [cej_seg_up], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(img, [cej_seg_low], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow("Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return JsonResponse({
            "tooth_data": tooth_data,
            "pbl_seg": pbl_seg.tolist(), 
            "cej_seg_up": cej_seg_up.tolist(),
            "cej_seg_low": cej_seg_low.tolist()
        })
    
    else:
        return render(request, "Diagnosis_main.html")




# def diagnosis_home(request):
#     if request.method == "POST": # 사용자가 자신의 컴퓨터에서 이미지를 선택한 후 웹에 업로드하여 post로 보냄

#         image_file = request.FILES["imgfile"] 
#         image_data = image_file.read()

#         # 이미지 데이터를 NumPy 배열로 변환합니다.
#         nparr = np.fromstring(image_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (size_x, size_y)) 

#         # 사용자의 이미지를 입력 이미지로 사용하여 인공지능 예측 수행
#         results = model_tooth.predict(img, conf=0.6)
#         results_pbl = model_pbl.predict(img) 
#         results_cej = model_cej.predict(img)

#         # 결과에서 상자 좌표 가져오기
#         for v in results[0]:
#             box_x, box_y, box_w, box_h = int(v.boxes.data[0][0]), int(v.boxes.data[0][1]), (
#                         int(v.boxes.data[0][2]) - int(v.boxes.data[0][0])), (
#                         int(v.boxes.data[0][3]) - int(v.boxes.data[0][1]))
            
#             # 치아 좌표 획득
#             segmentation_points = np.array(list(v.masks.xy))
#             mask = np.zeros((size_y, size_x), dtype=np.uint8)
#             cv2.fillPoly(mask, [np.array(segmentation_points, np.int32)], 255)
            
#             # 중심과 각도 계산
#             moments = cv2.moments(mask)
#             cx = int(moments['m10'] / moments['m00'])
#             cy = int(moments['m01'] / moments['m00'])
#             angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])

#             # 각도를 기준으로 두 점 계산
#             x1 = cx + np.cos(angle)
#             y1 = cy + np.sin(angle)
#             x2 = cx - np.cos(angle)
#             y2 = cy - np.sin(angle)

#             # 기울기와 y절편 계산
#             slope = (y2 - y1) / (x2 - x1)
#             intercept = y1 - slope * x1
            
#             # 축 그리기



#         return render(request, "Diagnosis_main.html")
        
#     else:
#         return render(request, "Diagnosis_main.html")

