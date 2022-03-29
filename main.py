import torch
from PIL import Image
from fastapi import FastAPI
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to the logo detection API'}


# Defining the model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

"""Loading the model for inference"""
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint = torch.load('logo_detection.model')
model.load_state_dict(checkpoint["state_dict"])
model.eval()

"""Preprocessing the images"""
from torchvision import transforms as T

def get_transform(train):
    transformations = []
    transformations.append(T.ToTensor())
    if train:
        transformations.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transformations)


def test(model, message):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = [get_transform(train=False)(Image.open(message).convert("RGB"))]
    model.to(device)
    images = list(image.to(device) for image in images)
    predictions = model(images) 
    correct_preds = {'boxes': [], 'scores': []}
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if(score>0.7):
            correct_preds['boxes'].append(box.cpu().detach().numpy())
            correct_preds['scores'].append(score.cpu().detach().numpy())
    
    for idx, box in enumerate(correct_preds['boxes']):
        im = images[0].permute(1,2,0).cpu().detach().numpy()[round(box[1]):round(box[3]), round(box[0]):round(box[2])]
        plt.imshow(im)
        plt.imsave(f'save_images/im_{idx}.png', im)
        
    return {"message": "Done : Saved the cropped portions in the save_images folder"}


@app.get('/logo_detection/')
async def detect_spam_query(message: str):
    # return {'message': f'Your path is {message}'}
    return test(model, message)
