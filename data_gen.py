# 전역변수 선언
import os
DATASET_DIR_PATH = "./dataset"
SAVE_IMAGES_DIR_PATH = os.path.join(DATASET_DIR_PATH, "images")
SAVE_LABELS_DIR_PATH = os.path.join(DATASET_DIR_PATH, "labels")
BG_IMG_DIR_PATH = "./seed_data/conveyor_img"
OBG_IMG_DIR_PATH = "./seed_data/product_imgs"
OBG_JSON_DIR_PATH = "./seed_data/product_json"
NAME_TO_ID = {"obj":0, "date":1}
NUM_DATA = 1000

# # 폴더 없으면 만들기
import os
if not os.path.exists(DATASET_DIR_PATH): os.makedirs(DATASET_DIR_PATH)
if not os.path.exists(SAVE_IMAGES_DIR_PATH): os.makedirs(SAVE_IMAGES_DIR_PATH)
if not os.path.exists(SAVE_LABELS_DIR_PATH): os.makedirs(SAVE_LABELS_DIR_PATH)

# 이미지 증축
# 사전에 seed_data에 데이터가 있어야함
from mytool import gen_data
gen_data(NAME_TO_ID, NUM_DATA)
