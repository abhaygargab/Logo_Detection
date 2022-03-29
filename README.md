# Logo_Detection
Train an Object Detection model for Logo Detection, also provide a script to use it as an API service.

# Setting Up the Environment
conda create -n new_env python=3.6
conda activate new_env
Pip install matplotlib
Pip install torch, torchvision
Pip install numpy
Pip install scikit-learn
Pip install scikit-image
pip install pycocotools
pip install "uvicorn[standard]"
pip install "fastapi[all]"


# Project Structure :
images (contains images to train the model)
save_images (to save the cropped images during inference)
downloaded (images used for data generation)
dataset.pkl : Generated Dataset
createDataset.py : Script to create the dataset
engine.py : Script containing training code
train.py : training script
main.py : File used to host model as API service
coco_eval.py (file used for training)
coco_utils.py (file used for training)

# Inference using trained model
1. Download the trained model from : "https://drive.google.com/file/d/1TM2-K4kh-B4OyQh6XBWq2-mFG5zEK7R1/view?usp=sharing"
2. conda activate new_env
3. Use command : uvicorn main:app --reload
4. Open localhost server on the browser: http://127.0.0.1:8000/logo_detection/?message="Path to image"
