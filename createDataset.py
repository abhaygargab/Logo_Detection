import os
import numpy as np
import random
import torch
import pickle
from PIL import Image

# Using 2 differnt images for two different types of logos of Allianz
img1 = Image.open("./downloaded/logo1.png", 'r')
new_img1 = img1.resize((100, 25))
img2 = Image.open("./downloaded/logo2.png", 'r')
new_img2 = img2.resize((100, 25))
img_w, img_h = new_img1.size

list_imgs = [new_img1, new_img2]
# dataset = {'images': [], 'bboxes': [], 'labels': []}

# Creating the dataset
dataset = []
for i in range(100):
    # Randomly choose on of the two types of Allinz logos for the document
    new_img = random.choice(list_imgs)
    # Using a random document to act as background on which we will add the logos
    background = Image.open('./downloaded/document.png').convert("RGB")
    bg_w, bg_h = background.size
    # As logos can be different sizes (maybe?)
    scale = random.uniform(0.7, 1)
    img_wd = round(scale*img_w)
    img_ht = round(scale*img_h)
    offset_x = random.randrange(img_wd, bg_w)
    offset_y = random.randrange(img_ht, 90-img_ht) # Add logo only in the empty space in the header
    x_top, y_top = (bg_w - offset_x), (bg_h - offset_y)
    background.paste(new_img, (x_top, y_top))
    offset_x = random.randrange(img_wd, bg_w)
    offset_y = random.randrange(800, bg_h-img_ht) # Add logo only in the empty space below the body
    x_bot, y_bot = (bg_w - offset_x), (bg_h - offset_y)
    background.paste(new_img, (x_bot, y_bot))    
    background.save(f'./images/image_{i}.png')
    dataset.append([f'./images/image_{i}.png', 
                    [[x_bot, y_bot, (x_bot+img_wd), (y_bot+img_ht)],
                     [x_top, y_top, (x_top+img_wd), (y_top+img_ht)]],
                    [1, 1]])    
    

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
