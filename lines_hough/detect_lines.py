import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# TODO: Find the best ROI for the camera
def roi(frame, (x1, x2), (y1, y2)):
    pass


while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            try:
                slope = float((y2 - y1)) / float((x2 - x1))
                print slope
            except ZeroDivisionError:
                print "Vetical"

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("image", img)
    q = ord('q') & 0xff
    if cv2.waitKey(2) == q:
        break

cap.release()
