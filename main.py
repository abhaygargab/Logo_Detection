import torch
from PIL import Image
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to the logo detection API'}


# Defining the model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint = torch.load('/home/u1698461/Documents/ImpPersonalDocs/send/al/logo_detection.model')
model.load_state_dict(checkpoint["model_state_dict"])

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
    images = list(image.to(device) for image in images)
    predictions = model(images) 
    correct_preds = {'boxes': [], 'scores': []}
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if(score>0.7):
            # correct_preds.append([box.cpu().detach().numpy(), score.cpu().detach().numpy()])
            correct_preds['boxes'].append(box.cpu().detach().numpy())
            correct_preds['scores'].append(score.cpu().detach().numpy())
    # label = model.predict([message])[0]
    # spam_prob = model.predict_proba([message])
    return {'boxes': correct_preds['boxes'], 'scores': correct_preds['scores']}


# @app.get('/logo_detection/')
# async def detect_spam_query(message: str):
#    return classify_message(model, message)


@app.get('/query_image_path/{message}')
async def detect_spam_path(message: str):
   return test(model, message)