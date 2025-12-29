from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import warnings

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

model_path = "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b"

prompt = "Suppose you are driving, and I'm providing you with six images captured by the car's front, front-left, front-right, back, back-left and back-right camera. First, generate a description of the driving scene which includes the key factors for driving planning, including the presence of obstacles and the positions and movements of vehicles and pedestrians and traffic lights. After description, please predict the behavior of ego vehicle, including exactly the driving direction(straight, turn left or turn right) and driving speed(slow, fast or normal)."

# prompt = "Describe the image in detail."

image_file = "/root/fsas/dataset/OpenDriveLab/DriveLM/val_data/CAM_FRONT/n008-2018-09-18-14-35-12-0400__CAM_FRONT__1537295999112404.jpg,/root/fsas/dataset/OpenDriveLab/DriveLM/val_data/CAM_FRONT_LEFT/n008-2018-09-18-14-35-12-0400__CAM_FRONT_LEFT__1537295999104799.jpg,/root/fsas/dataset/OpenDriveLab/DriveLM/val_data/CAM_FRONT_RIGHT/n008-2018-09-18-14-35-12-0400__CAM_FRONT_RIGHT__1537295999120482.jpg,/root/fsas/dataset/OpenDriveLab/DriveLM/val_data/CAM_BACK/n008-2018-09-18-14-35-12-0400__CAM_BACK__1537295999137558.jpg,/root/fsas/dataset/OpenDriveLab/DriveLM/val_data/CAM_BACK_LEFT/n008-2018-09-18-14-35-12-0400__CAM_BACK_LEFT__1537295999147405.jpg,/root/fsas/dataset/OpenDriveLab/DriveLM/val_data/CAM_BACK_RIGHT/n008-2018-09-18-14-35-12-0400__CAM_BACK_RIGHT__1537295999128113.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": "llava-v1.6-vicuna-7b", 
    "query": prompt,
    "conv_mode": "llava_v1", 
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()


eval_model(args)