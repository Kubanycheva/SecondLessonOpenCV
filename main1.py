import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
if not cap:
    print('Camera not found')
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    fps_start = time.time()

    ret, frame = cap.read()
    if not ret:
        print('Frame not found')
        break

    result = model(frame, conf=0.3)

    boxes = result[0].boxes
    person_count = 0

    for n in boxes:
        cls = int(n.cls[0])
        label = model.names[cls]
        conf = round(float(n.conf[0]), 2)

        if label == 'person':
            person_count += 1

            x, y, w, h = map(int, n.xyxy[0])
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}, {conf * 100}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            fps_end = time.time()
            fps = 1 / (fps_end - fps_start)
            cv2.putText(frame, f'Fps:{round(fps, 1)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, f'Person count: {person_count}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    cv2.imshow('Video: ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()