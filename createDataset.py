import os
import numpy as np
import random
import torch
import pickle
from PIL import Image

img_path = "ss.png"
img = Image.open(img_path).convert("RGB")

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(img)
plt.show()

plt.figure()
plt.imshow(img.resize((100, 25)))
plt.show()


img = Image.open("ss.png", 'r')
new_img = img.resize((100, 25))
img_w, img_h = new_img.size

# dataset = {'images': [], 'bboxes': [], 'labels': []}
dataset = []
for i in range(100):
    background = Image.new('RGBA', (600, 800), (255, 255, 255, 255))
    bg_w, bg_h = background.size
    offset_x = random.randrange(img_w, bg_w)
    offset_y = random.randrange(img_h, bg_h)
    x, y = (bg_w - offset_x), (bg_h - offset_y)
    background.paste(new_img, (x, y))
    background.save(f'./images/image_{i}.png')
    dataset.append([f'./images/image_{i}.png', 
                    [x, y, (x+img_w), (y+img_h)],
                    1])
    

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
