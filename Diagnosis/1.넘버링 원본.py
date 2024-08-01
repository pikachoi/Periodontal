def lineDetection(self):
    try:
        self.btn_test.setChecked(False)
        if len(globals()['df_line']) == 0 and len(self.current_img) != 0:
            df_dentex = pd.DataFrame(columns=['id', 'segmentation', 'center_point'])
            detect_img = cv2.resize(self.img, (self.size_x, self.size_y))
            detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB, detect_img)

            results_quardrant = self.model_quardrant.predict(detect_img, conf=0.6)
            for result in results_quardrant[0]:
                area_id = str(int(result.boxes.cls.tolist()[0]))
                segmentation_points = list(map(lambda x: [int(x[0]), int(x[1])], list(result.masks.xy)[0].tolist()))
                mask = np.zeros((self.size_y, self.size_x), dtype=np.uint8)
                cv2.fillPoly(mask, np.array([segmentation_points], dtype=np.int32), 255)
                mask_inv = cv2.bitwise_not(mask)
                padding_image = np.full((self.size_y, self.size_x, 3), (0, 0, 0), dtype=np.uint8)
                image_padded = cv2.bitwise_and(padding_image, padding_image, mask=mask_inv)
                globals()[f'image_padded_{area_id}'] = cv2.add(image_padded, cv2.bitwise_and(detect_img, detect_img, mask=mask))
                results_dentex = self.model_dentex.predict(globals()[f'image_padded_{area_id}'], conf=0.6)
                for result_dentex in results_dentex[0]:
                    box_x, box_y, box_w, box_h = int(result_dentex.boxes.data[0][0]), int(result_dentex.boxes.data[0][1]), (
                            int(result_dentex.boxes.data[0][2]) - int(result_dentex.boxes.data[0][0])), (
                            int(result_dentex.boxes.data[0][3]) - int(result_dentex.boxes.data[0][1]))
                    tooth_id = str(int(result_dentex.boxes.cls.tolist()[0]))
                    segmentation_points = list(map(lambda x: [int(x[0]), int(x[1])], list(result_dentex.masks.xy)[0].tolist()))
                    df_dentex.loc[len(df_dentex)] = {'id': f'{area_id}_{tooth_id}', 'segmentation': segmentation_points, 'center_point': [int(box_x + box_w / 2), int(box_y + box_h / 2)]}

            self.results = self.model_tooth.predict(detect_img, conf=0.6)
            center_list = []
            for v in self.results[0]:
                box_x, box_y, box_w, box_h = int(v.boxes.data[0][0]), int(v.boxes.data[0][1]), (
                            int(v.boxes.data[0][2]) - int(v.boxes.data[0][0])), (
                            int(v.boxes.data[0][3]) - int(v.boxes.data[0][1]))
                center_list.append([int(box_x + box_w / 2), int(box_y + box_h / 2)])
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
                globals()['df_line'].loc[len(globals()['df_line'])] = {'id' : self.class_list[int(v.boxes.cls.tolist()[0])], '시작 좌표' : axis_line[0], '종료 좌표' : axis_line[1]}

            for i, row in df_dentex.iterrows():
                dist_list = list(map(lambda x: self.distance(row['center_point'][0], row['center_point'][1], x[0], x[1]), center_list))
                globals()['df_line']['id'][dist_list.index(min(dist_list))] = row['id']

            self.append_table('line')
            self.drawImg('total')
    except:
        pass