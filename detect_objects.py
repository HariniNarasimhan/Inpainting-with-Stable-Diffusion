# import necessary libraries
# %matplotlib inline
from PIL import Image
# import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings
warnings.filterwarnings('ignore')

# load model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# set to evaluation mode
model.eval()

# load COCO category names
COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_coloured_mask(mask):
  """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)
  r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def get_prediction(img_path, confidence):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - confidence - threshold to keep the prediction or not
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0
    
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  # print(pred[0]['labels'].numpy().max())
  pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class


class Detectobjects():
    def __init__(self, img_path):
        self.img_path = img_path
        masks, boxes, pred_cls = get_prediction(img_path, 0.5)
        self.masks = masks
        self.boxes = boxes
        self.pred_cls = pred_cls

    def save_predictions(self, save_path):
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(self.masks)):
            rgb_mask = get_coloured_mask(self.masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            x1,y1= (int(self.boxes[i][0][0]),int(self.boxes[i][0][1]))
            x2,y2 = (int(self.boxes[i][1][0]),int(self.boxes[i][1][1]))
            center = (x1+int((x2-x1)/2), y1+int((y2-y1)/2))
            cv2.circle(img, (x1+int((x2-x1)/2)+3, y1+int((y2-y1)/2)-3), 15, (0,0,0), -1)
            cv2.putText(img,str(i),center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),thickness=2)

        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    def fetch_predictions(self, indices, save_path):
        result_masks= [self.masks[i] for i in range(len(self.masks)) if i not in indices ] 
        combined_mask = np.clip(sum(result_masks), 0, 1)
        cv2.imwrite(save_path, combined_mask*255)
        return result_masks

def resize(orig_img_path, orig_mask_path, img_path, mask_path):
  im = cv2.imread(orig_img_path)
  mask = cv2.imread(orig_mask_path,0)
  aspect_ratio = im.shape[0]/im.shape[1]
  if im.shape[0] > im.shape[1]:
    w = (int(512 / aspect_ratio))//64
    size = (64*w,512)
  else:
    h = (int(512* aspect_ratio))//64
    size = (512, 64*h)
  cv2.imwrite(img_path, cv2.resize(im, size))
  cv2.imwrite(mask_path, cv2.resize(mask, size))



if __name__=='__main__':
  import os

  os.makedirs('input_dir2', exist_ok=True)
  orig_img_path = '/home/ec2-user/stable-diffusion/stable-diffusion/tajmahal.jpg'
  detect = Detectobjects(orig_img_path)
  det_path = '/home/ec2-user/stable-diffusion/stable-diffusion/tajmahal-detect.jpg'
  detect.save_predictions(det_path)
  orig_mask_path = '/home/ec2-user/stable-diffusion/stable-diffusion/tajmahal_mask.jpg'
  detect.fetch_predictions([0,4,1,5],orig_mask_path)
  img_path = 'input_dir/tajmahal.png'
  mask_path = 'input_dir/tajmahal_mask.png'
  resize(orig_img_path, orig_mask_path, img_path, mask_path)


        






