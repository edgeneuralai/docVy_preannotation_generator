import os
import glob
from uuid import uuid4
from PIL import Image
import json
import os
import torch
from pdf2image import convert_from_path
from model import PPOCRModelWrapper

INPUT_DIR = os.environ["INPUT_DIR"]
split_count = int(input("Enter number of splits:"))

ocr = PPOCRModelWrapper()

for i, dir in enumerate(glob.glob(os.path.join(INPUT_DIR, "*"))):
    if not os.path.isdir(dir) :
        continue

    tasks = []

    FOLDER_PATH = os.path.join(INPUT_DIR, dir)

    PDFS = glob.glob(os.path.join(FOLDER_PATH, "*.pdf"))
    
    PREANNOTATION_DIR = os.path.join(FOLDER_PATH, "preannotations")
    IMAGES_DIR = os.path.join(FOLDER_PATH, "images")
    
    if not os.path.isdir(PREANNOTATION_DIR):
        os.makedirs(PREANNOTATION_DIR)
    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    for pdf_path in PDFS :
        for i, image in enumerate(convert_from_path(pdf_path)):
            image.save(os.path.join(IMAGES_DIR,f"{i:04d}.jpg"))

    IMAGES = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))

    for image_path in IMAGES :
        base_path = image_path.split("/")[-1]
        out = {
            "data":{
                "ocr":f"http://localhost:8000/{base_path}"
            },
            "predictions":[
                {
                    "result":[],
                    "score":0.98
                }
            ]
        }

        pil_image = Image.open(image_path)

        result = ocr.batch_predict(
            [pil_image]
        )[0]
        #print(result)
        width, height = pil_image.size
        pil_image.close()
        for box, text in zip(json.loads(result["PageJson"])["word_bbox"], json.loads(result["PageJson"])["words"]):

           x0,y0,x1,y1 = box
                
           w = x1-x0
           h = y1-y0
   
           bbox = {
               "x": 100* x0 ,#/ width,
               "y": 100* y0 ,#/ height,
               "width":  100 * w,# / width, 
               "height": 100 * h,#/ height, 
               "rotation":0}
   
           region_id = str(uuid4())[:10]
           score = 0.98
   
           bbox_result = {
                   'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
                   'value': bbox}
           transcription_result = {
                   'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
                   'value': dict(text=[text], **bbox), 'score': score}
   
           out["predictions"][0]["result"].extend([bbox_result, transcription_result])
           #print(out)
           #print("!!!!!!!!!")
        tasks.append(out)

    #os.system(f"rm {FOLDER_PATH}/*.json")
    
    if split_count > len(tasks) :
        print("split count greater than task count, number of splits willl be less")
        split_count = len(tasks)
    
    task_chunk_size = len(tasks)//split_count
    for i in range(split_count) :
        start = i*task_chunk_size
        end = (i+1)*task_chunk_size
        if end > len(tasks) :
            end = -1
                
        file_writer = open(os.path.join(PREANNOTATION_DIR, f"preannotation_split_{i}.json"), 'w')
        file_writer.write(json.dumps(tasks[start:end], indent=4))
        file_writer.flush()
        file_writer.close()

    print(f"Preannotation for {i+1} classes generated, Generating rest")
print("Generated All Preannotations")