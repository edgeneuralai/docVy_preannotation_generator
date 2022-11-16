import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import PaddleOCR.tools.infer.utility as utility
import PaddleOCR.tools.infer.predict_rec as predict_rec
import PaddleOCR.tools.infer.predict_det as predict_det
import PaddleOCR.tools.infer.predict_cls as predict_cls
from PaddleOCR.ppocr.utils.utility import get_image_file_list, check_and_read
from PaddleOCR.ppocr.utils.logging import get_logger
from PaddleOCR.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
import argparse

logger = get_logger()

def str2bool(v):
    return v.lower() in ("true", "t", "1")

class TextSystem(object):

    def __init__(self, config):
        args = self.init_args(config)
        if not args.show_log:
            logger.setLevel(logging.INFO)
        
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def init_args(self,config):
        parser = argparse.ArgumentParser()
        # params for prediction engine
        use_gpu= False if config.device=="cpu" else True
        parser.add_argument("--use_gpu", type=str2bool, default=use_gpu)
        parser.add_argument("--use_xpu", type=str2bool, default=False)
        parser.add_argument("--use_npu", type=str2bool, default=False)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--min_subgraph_size", type=int, default=15)
        parser.add_argument("--precision", type=str, default="fp32")
        parser.add_argument("--gpu_mem", type=int, default=500)

        # params for text detector
        parser.add_argument("--det_algorithm", type=str, default='DB')
        det_model_dir = config.TEXT_DETCTION_MODEL
        parser.add_argument("--det_model_dir", type=str,default=det_model_dir)
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')
        parser.add_argument("--det_box_type", type=str, default='quad')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
        parser.add_argument("--max_batch_size", type=int, default=10)
        parser.add_argument("--use_dilation", type=str2bool, default=False)
        parser.add_argument("--det_db_score_mode", type=str, default="fast")

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # SAST parmas
        parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
        parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)

        # PSE parmas
        parser.add_argument("--det_pse_thresh", type=float, default=0)
        parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
        parser.add_argument("--det_pse_min_area", type=float, default=16)
        parser.add_argument("--det_pse_scale", type=int, default=1)

        # FCE parmas
        parser.add_argument("--scales", type=list, default=[8, 16, 32])
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--fourier_degree", type=int, default=5)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
        rec_model_dir = config.TEXT_RECOGNITION_MODEL
        parser.add_argument("--rec_model_dir", type=str,default=rec_model_dir)
        parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
        parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
        parser.add_argument("--rec_batch_num", type=int, default=6)
        parser.add_argument("--max_text_length", type=int, default=25)
        rec_char_dict_path = config.REC_CHAR_DICT_PATH
        parser.add_argument(
            "--rec_char_dict_path",
            type=str,
            default=rec_char_dict_path)
        parser.add_argument("--use_space_char", type=str2bool, default=True)
        parser.add_argument(
            "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
        parser.add_argument("--drop_score", type=float, default=0.5)

        # params for e2e
        parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
        parser.add_argument("--e2e_model_dir", type=str)
        parser.add_argument("--e2e_limit_side_len", type=float, default=768)
        parser.add_argument("--e2e_limit_type", type=str, default='max')

        # PGNet parmas
        parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
        parser.add_argument(
            "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
        parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
        parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

        # params for text classifier
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)
        parser.add_argument("--cls_model_dir", type=str)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=6)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
        parser.add_argument("--cpu_threads", type=int, default=10)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)
        parser.add_argument("--warmup", type=str2bool, default=False)

        # SR parmas
        parser.add_argument("--sr_model_dir", type=str)
        parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
        parser.add_argument("--sr_batch_num", type=int, default=1)

        #
        parser.add_argument(
            "--draw_img_save_dir", type=str, default="./inference_results")
        parser.add_argument("--save_crop_res", type=str2bool, default=False)
        parser.add_argument("--crop_res_save_dir", type=str, default="./output")

        # multi-process
        parser.add_argument("--use_mp", type=str2bool, default=False)
        parser.add_argument("--total_process_num", type=int, default=1)
        parser.add_argument("--process_id", type=int, default=0)

        parser.add_argument("--benchmark", type=str2bool, default=False)
        parser.add_argument("--save_log_path", type=str, default="./log_output/")

        parser.add_argument("--show_log", type=str2bool, default=True)
        parser.add_argument("--use_onnx", type=str2bool, default=False)
        return parser.parse_args()

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes