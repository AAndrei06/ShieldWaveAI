import cv2
import imutils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width = 500)
start_frame = cv2.cvtColor(start_frame,cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21,21), 0)
i = 0
while True:
    _, frame = cap.read()

    frame = imutils.resize(frame, width = 500)
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_bw = cv2.GaussianBlur(frame_bw, (5,5), 0)
    difference = cv2.absdiff(frame_bw, start_frame)

    threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
    start_frame = frame_bw

    if threshold.sum() > 100000:
        print(f'ALERT-----{i}')

    cv2.imshow('CAM', frame)
    cv2.imshow('THRESh', threshold)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()