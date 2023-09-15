import cv2
import numpy as np
import torch
import onnxruntime
import time
import os
from PIL import Image
from torchvision import transforms
from skimage.transform import resize
from P3M_net import P3M_net
import tqdm

onnx_P3M = '/root/zhangpeng-chengdu/matting/P3M-Net-main/ckpt_epoch1.onnx'  # onnx 模型的路径
orig_path = '/root/zhangpeng-chengdu/matting/P3M-Net-main/samples/original/'
alp_path  = '/root/zhangpeng-chengdu/matting/P3M-Net-main/samples/result_alpha/'
col_path = '/root/zhangpeng-chengdu/matting/P3M-Net-main/samples/result_color/'
#device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')  # 判断是用GPU 还是cpu
device = 'cpu'
P3M = P3M_net(model_path = onnx_P3M , device = device)

for filename in tqdm.tqdm(os.listdir(orig_path)):
    img_name = filename.split('.')[0]
    img_path = orig_path + img_name + '.jpg'
    img , predict = P3M.infer(img_path)
    col_img  =  P3M.generate_composite_img(img , predict)

    col = col_path + img_name + '.png'
    cv2.imwrite(col,col_img)

    mask_path = alp_path + img_name + '.png'
    predict = predict * 255.0
    cv2.imwrite(mask_path, predict.astype(np.uint8))













