# 전역변수 선언
SOURCE_PATH = "./temp/shot.jpg"
WEIGHTS_PATH = "./yolov5/yolov5s.pt"

# 소켓서버에 연결
from mytool import connection_for_client
import time
time.sleep(1)
client = connection_for_client() # 먼저 서버쪽이 켜져있어야함

# 감지한것을 텍스트로
def detect_to_txt(detection, img): # 탐지한 날짜를 리턴
    detection = detection.numpy()
    ids = detection[:,-1]
    boxes = detection[:,:4]
    for id, box in zip(ids, boxes):
        if names[int(id)] == "date":
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            crop_img = cv2.adaptiveThreshold(crop_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
            text = pytesseract.image_to_string(crop_img, config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.")
            return text
    else: return None

# 모델 실행
import cv2
from yolov5.detect import run
import time

gen = run(source=SOURCE_PATH, weights=WEIGHTS_PATH, client=client, view_img=True, nosave=True)
names = next(gen) # 객체 이름 리스트

print("client >> gogo ", end='')
client.send(b"gogo")

# 서버로부터 아무값을 받으면 이미지탐지 실행
# 15초동안 못받으면 종료
for img, detection in gen:
    text = detect_to_txt(detection, img)
    if text:
        pass # 서버로 보내기?

cv2.destroyAllWindows()

