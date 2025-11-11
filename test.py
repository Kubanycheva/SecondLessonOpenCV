import cv2
import datetime
import os
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO('yolov8n.pt')

video_path = 'videos'
if not os.path.exists(video_path):
    os.makedirs(video_path)


image_path = 'media'
if not os.path.exists(image_path):
    os.makedirs(image_path)


if not cap.isOpened():
    print('Camera not found')
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

if frame_fps == 0:
    frame_fps = 30.0

date = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_name = f'{video_path}/video_{date}.mp4'
out = cv2.VideoWriter(video_name, fourcc, frame_fps, (frame_width, frame_height))



while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame not found')
        break

    result = model(frame, stream=True, conf=0.5)

    for i in result:
        for n in i.boxes:
            cls = int(n.cls[0])
            label = model.names[cls]
            conf = round(float(n.conf[0]), 2)

            x, y, w, h = map(int, n.xyxy[0])
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}, {conf*100}%', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv2.putText(frame, 'Hello World', (100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Videos: ', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        print('Запись башталды')
        out.write(frame)
    elif key == ord('s'):
        print('Скриншот болду')
        image_file = f'{image_path}/photo_{date}.jpg'
        cv2.imwrite(image_file, frame)


cap.release()
out.release()
cv2.destroyAllWindows()
