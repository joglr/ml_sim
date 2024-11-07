import cv2
import numpy as np

camera = cv2.VideoCapture(0)

cv2.startWindowThread()

def end():
  cv2.destroyAllWindows()
  camera.release()

while True:
    ret, frame = camera.read()
    if not ret:
        print("failed to grab frame")
        end()
        break

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # resize image

    lower_color = np.array([20, 60, 50])
    upper_color = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blurred_frame = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(blurred_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 4000:
            print("Detected area of ", area)
            print(contour)
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

    cv2.imshow("Video", blurred_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        end()
        break
