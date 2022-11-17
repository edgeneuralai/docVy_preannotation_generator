#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#import torch

import io
import logging

import numpy as np
import os
from PIL import Image
import json
#from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta
import config 
from config import *
import config
logger = logging.getLogger()

from ppocr_predict import TextSystem
from paddleocr import PaddleOCR,draw_ocr


class PPOCRModelWrapper:

    def __init__(self, path=""):
        from paddleocr import PaddleOCR,draw_ocr
        logger.info("Loading PPOCR...")
        # Initialize the OCR
        self.ppocr = PaddleOCR(
            use_angle_cls = False,
            lang='en',
            det_model_dir = "ppocr_models/det_db",
            use_gpu= False if config.device=="cpu" else True
        )

        logger.info('Loaded model')

    def _read_image(self, image_data):
        '''Read the image from a Bytestream.'''
        images = pdf2image.convert_from_bytes(image_data, thread_count=16)
        #image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return images

    def _pre_process(self, image):
        image = np.asarray(image)
        return image

    def _predict(self, image):
        '''Call the model'''
        raw_output = self.ppocr.readtext(
            image,details=True
        )
        return raw_output
    
    def batch_predict(self, images):

        images = [np.asarray(x.convert("RGB")) for x in images]
        
        print("Length of images = ", len(images))
        out = []
        for image in images :
            raw_output = self.ppocr.ocr(
                image, cls=False
            )
            height, width, channel = image.shape
            words = []
            boxes = []
            confidences = []

            page_json = {
                "PageOrientation":"PAGE_UP",
                "Height":height,
                "Width":816
            }

            for token_info in raw_output[0] :
                raw_box = token_info[0]
                x0 = min([x[0] for x in raw_box]) / width
                y0 = min([x[1] for x in raw_box]) / height
                x1 = max([x[0] for x in raw_box]) / width
                y1 = max([x[1] for x in raw_box]) / height
                boxes.append([x0, y0, x1, y1])
                words.append(token_info[1][0])
                confidences.append(token_info[1][1])
            
            inner_json=dict()
            inner_json["words"]=words
            inner_json["word_bbox"]=boxes
            inner_json["word_conf"]=confidences
            inner_json["blank"]=not(len(words)>0)
            inner_json["height"]=height
            inner_json["width"]=width
            
            page_json["PageText"]=" ".join(words)
            page_json["PageJson"] = json.dumps(inner_json)
            out.append(page_json)
        return out

    def write_image(self, result):
        '''Return the generated image as output.'''
        return 0

    def _post_process(self, result):
        '''Post-processing.'''
        return 

if __name__=="__main__":
    from PIL import Image
    ppocr_engine = PPOCRModelWrapper()
    results = ppocr_engine.batch_predict(
        [
            Image.open(
                "/home/captain-america/doc-understanding-deployment/test.jpg"
            )
        ]
    )
    print(results)