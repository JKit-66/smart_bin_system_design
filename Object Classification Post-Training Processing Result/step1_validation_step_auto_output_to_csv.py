import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from collections import Counter
import time
import uuid

model = YOLO('path/to/best.pt')  #v7_18_3/best.pt

classNames = ['shoe', 'paperBox', 'pastry', 
              'penPencil', 'milkCarton', 'cutlery', 
              'crumpledPaper', 'eggShell', 'glassBottle', 
              'plasticContainer','paperEnvelope', 'paperCup', 
              'fruit', 'noodlePasta', 'plasticLid', 
              'plasticMilkBottle', 'sandwich','tissueCore', 
              'vape', 'vegeScraps', 'glassJar', 
              'meat', 'book', 'plasticBottle', 
              'alCan', 'softPlastic']

path_now = Path().resolve()
main_file = 'dataset2'
main_result = 'results'
main_directory = os.path.join(path_now, main_file) 

columns = ["Idx", "Class", "Valid FileName", "Result FileName", 
           "Number of Item Detected", "Detected Classes Idx", 
           "Detected Classes Name", "Confidence Score"]
items = []
all_sim_time = []

idx = 0
all_class = os.listdir(main_directory) 
start_time = time.time()
for each_class in all_class:
    per_class = os.path.join(main_directory, each_class) 
    if os.path.isdir(per_class):
        # Create the corresponding folder in 'results' if it doesn’t exist
        result_class_dir = os.path.join(path_now, main_result, each_class)  # e.g., C:\...\results\shoe
        os.makedirs(result_class_dir, exist_ok=True)
        all_image_each_cls = os.listdir(per_class)
        idx += 1
        for each_img in all_image_each_cls:
            each_img_dir = os.path.join(per_class, each_img)
            img = cv2.imread(each_img_dir)
            results = model(each_img_dir)

            for i in results:
                boxes = i.boxes
                
                for box in boxes:
                    #confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    
                    if conf > 0.2:       
                        #Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                        
                        w, h = x2-x1, y2-y1
                        bbox = (x1, y1, w, h)
                        cvzone.cornerRect(img, bbox)
                        
                        
                        #print(conf)
                        
                        #class name
                        cls = int(box.cls[0])
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(40,y1)))
                        #print(classNames[cls])
            
            unique_id = uuid.uuid4().hex 
            file_res_name = f'{Path(each_img).stem}_{unique_id}.jpg'
            output_path = os.path.join(path_now,main_result,each_class,file_res_name)
            cv2.imwrite(output_path, img)

            num_detected = len(results[0].boxes)
            each_img_detected = results[0].boxes.cls.tolist()
            each_img_conf = results[0].boxes.conf.tolist()
            result_clss_name = [classNames[int(i)] for i in each_img_detected]
            items.append([idx, each_class, each_img, file_res_name, num_detected, each_img_detected, result_clss_name, each_img_conf])

        end_time = time.time()
        #print(f' Tested done with: {init_num} dataset')
        elapsed_time = end_time - start_time
        print(f' Simulaiton done within: {elapsed_time} time (s) for {each_class}')
        all_sim_time.append(elapsed_time)

df = pd.DataFrame(items, columns=columns)
main_excel_directory = os.path.join(path_now, f'DetectionResults_{unique_id}.csv')
                         
df.to_csv(main_excel_directory, index=False)

end_time_all = time.time()
elapsed_time_all = end_time_all - start_time

print(f'### Total Simulation time is: {all_sim_time}')
print(f'#### VALIDATION done for all within: {elapsed_time_all} time (s)')
