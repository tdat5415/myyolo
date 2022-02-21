# 전역변수 선언하기
import os
myname = "webtext"
root_path = os.getcwd()
py_path = os.path.join(root_path, "yolov5/train.py")
data_yaml_path = os.path.join(root_path, "dataset/data.yaml")
yolo_yaml_path = os.path.join(root_path, "yolov5/models/yolov5.yaml")
pt_path = os.path.join(root_path, f"yolov5/runs/train/{myname}/weights/best.pt")

# 이미지 경로 텍스트파일 만들기
from glob import glob
import os
from sklearn.model_selection import train_test_split

img_list = sorted(glob(os.path.join(root_path, 'dataset/images/*.jpg')))
train_img_list, test_img_list = train_test_split(img_list, test_size=0.1, random_state=0)
train_img_list, valid_img_list = train_test_split(train_img_list, test_size=0.2, random_state=0)

with open(os.path.join(root_path, 'dataset/train.txt'), 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open(os.path.join(root_path, 'dataset/valid.txt'), 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')

# yaml 파일 수정하기
import yaml

with open(os.path.join(root_path, 'dataset/data.yaml'), 'r') as f:
    data = yaml.load(f, yaml.FullLoader)

data['names'] = ["obj", "date"]
data['nc'] = len(data['names'])
data['train'] = os.path.join(root_path, 'dataset/train.txt')
data['val'] = os.path.join(root_path, 'dataset/valid.txt')

with open(os.path.join(root_path, 'dataset/data.yaml'), 'w') as f:
    yaml.dump(data, f)
    
print("Completed editing yaml file.")

# 학습하기
import subprocess
img_size = 640
batch = 16
epochs = 50
command = f"python {py_path} --img {img_size} --batch {batch} --epochs {epochs} --data {data_yaml_path} --cfg {yolo_yaml_path} --weights {pt_path} --name {myname}"
subprocess.run(command, shell=True)





