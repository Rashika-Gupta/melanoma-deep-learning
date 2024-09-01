# upload image
#edpoint to upload image uing flask 

# save image
#function to make prediction on image
#show the result
import os
import torch

import albumentations # 
import pretrainedmodels

import numpy as np 
import torch.nn as nn

from flask import Flask 
from flask import render_template, request, jsonify
from torch.nn import functional as F

from wtfml.data_loaders.image import ClassificationLoader#wtfml 

app = Flask(__name__)
# let there be global variable upload folder and say it that upload it in static folder
UPLOAD_FOLDER = "~/melanoma-deep-learning/static"
DEVICE = "cuda"
MODEL = None
# model class
class SEResNext50_32x4d(nn.Module) :
    def __init__(self, pretrained = "imagenet"): #load pretrained weifgt
        super(SEResNext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained = pretrained) # doesn't load weight if not pretrained
        # metric area - roc can take line output
        self.out = nn.Linear( 2048, 1)# change last layer or use features from model.. 

    def forward(self, image,target):
        # batchsizze, channel, height, width
        bs, _,_,_ = image.shape
        #images are passed 
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x,1)
        # reshpa to batch shize
        x = x.reshpa(bs, -1)
        # paaas it through output layere
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(
            out, target.view(-1,1).type_as(out)
        )
        return out, loss

# predict function (need image path)
def predict(image_path, model):

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply = True)
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationDataset(
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis = 0)
    image = image / 255.0
    image = image.transpose((0, 3, 1, 2))
    image = torch.tensor(image, dtype = torch.float).to(DEVICE)
    with torch.no_grad():
        model.eval()
        outputs = model(image)
    return outputs
    

@app.route("/", methods = ["GET", "POST"]) # get request and post request

def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"] # get the image file from the request where "image" is the name provided in htmll. will be only there if reques method is post
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            print(pred)
            return render_template("index.html", prediction = 1)
    return render_template("index.html", prediction = 0) # when you render a template can send an argyment prediction = = and, in index.html you can use <h3>Prediction : {{prediction}}</h3.this prediction variable to display

if __name__ == "__main__":
    app.run(port = 12000, debug = True) # will show error if  there are any error in browser
   