import cv2
import sys

cap = cv2.VideoCapture('./movies/232538_tiny.mp4')

if not cap.isOpened():
    print('동영상을 불러올 수 없음')
    sys.exit()

print('동영상을 불러올 수 있음')

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(width)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(height)

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(frame_count)

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break

cap.release()

