import cv2
import numpy as np
import torch
import onnxruntime
import time
import os
from PIL import Image
from torchvision import transforms
from skimage.transform import resize

class P3M_net():
    def __init__(self,model_path,device):
        self.SHORTER_PATH_LIMITATION = 1080
        self.MAX_SIZE_H = 1600
        self.MAX_SIZE_W = 1600
        self.MIN_SIZE_H = 512
        self.MIN_SIZE_W = 512
        self.model = model_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.model, providers=self.providers)

    def infer(self,img):
        # img = np.array(Image.open(img_path))[:, :, :3]
        img = np.array(img)[:,:,:3]
        h, w , c = img.shape
        if min(h , w) > self.SHORTER_PATH_LIMITATION:
            if h >= w :
                new_w = self.SHORTER_PATH_LIMITATION
                new_h = int(self.SHORTER_PATH_LIMITATION * h / w)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                new_h = self.SHORTER_PATH_LIMITATION
                new_w = int(self.SHORTER_PATH_LIMITATION * w / h)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        h, w, c = img.shape
        h_org, w_org = h , w
        resize_h = int(h / 2)
        resize_w = int(w / 2)
        new_h = min(self.MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(self.MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w)) * 255.0
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)

        input_t = tensor_img
        input_t = input_t / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input_t = normalize(input_t)
        input_t = input_t.unsqueeze(0)

        input_t = np.array(input_t)
        input_t = input_t[0, :, :, :]

        ort_input = {self.session.get_inputs()[0].name: input_t[None, :, :, :]}
        pred_global, pred_local, pred_fusion = self.session.run(None, ort_input)[:3]
        pred_fusion = resize(pred_fusion[0, 0, :, :], (h_org, w_org))
        img = cv2.resize(img, (w_org, h_org), interpolation=cv2.INTER_LINEAR)
        return img , pred_fusion

    def generate_composite_img(self , img, alpha_channel):
        r_channel, g_channel, b_channel = cv2.split(img)
        b_channel = b_channel * alpha_channel
        g_channel = g_channel * alpha_channel
        r_channel = r_channel * alpha_channel
        alpha_channel = (alpha_channel * 255).astype(b_channel.dtype)
        img_BGRA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
        return img_BGRA

    def add_color(self,img,pred_fusion,color):

        cols, rows = img.shape[0], img.shape[1]
        back = np.zeros((cols, rows, 3), np.uint8)
        back[:] = color
        mask = pred_fusion
        # scenic_mask = ~mask
        scenic_mask = mask* 255.0
        scenic_mask = scenic_mask.astype(np.uint8)
        scenic_mask = ~scenic_mask
        scenic_mask = scenic_mask / 255.0
        back[:, :, 0] = back[:, :, 0] * scenic_mask
        back[:, :, 1] = back[:, :, 1] * scenic_mask
        back[:, :, 2] = back[:, :, 2] * scenic_mask

        img[:, :, 0] = img[:, :, 0] * mask
        img[:, :, 1] = img[:, :, 1] * mask
        img[:, :, 2] = img[:, :, 2] * mask
        result = cv2.add(back, img)
        return result















