"""
Inspired from https://github.com/ecm200/caltech_birds for ZooHackathon Europe 2020
"""

import torchvision.transforms as transforms
import os
import torch
from PIL import Image 

root_dir = '/content/drive/My Drive/ZooHackathon/CUB_200_2011/CUB_200_2011/'
model_file = os.path.join(root_dir, 'trainedModel.pth')
trainedModel = torch.load(model_file, map_location=torch.device('cpu'))

def run_prediction(input):
  input = preprocess(input).unsqueeze_(0)
  trainedModel.eval()
  input = input
  outputs = trainedModel(input)
  _, preds = torch.max(outputs, 1)
  p = preds.cpu().numpy()
  return p

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

'''
Example run

test_img = Image.open('/content/drive/My Drive/ZooHackathon/CUB_200_2011/CUB_200_2011/new_images/test/004.Groove_billed_Ani/Groove_Billed_Ani_0078_1780.jpg')
pred = run_prediction(test_img)
'''

