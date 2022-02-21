from threading import Thread
from mytool import connection_for_server
import subprocess
import time

# detect 실행과 동시에 소켓연결
t = Thread(target=lambda x:subprocess.run(x, shell=True), \
           args=("python ./yolo_detect.py",))
t.start()
with_client = connection_for_server()

# 모델 로드 기다리기
with_client.settimeout(10)
sk_data = with_client.recv(256)
print(f">> {sk_data} >> server")

# 촬영프로세스
### 


#####################
for i in range(5):
    print("server >> hello ", end='')
    with_client.send(b'hello')
    time.sleep(2)

print("server : finish.")
t.join()
