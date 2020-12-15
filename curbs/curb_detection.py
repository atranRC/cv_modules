import cv2
import numpy as np
from tracker import LaneTracker


# ym_per_pix = 27 / h  # meters per pixel in y dimension
# xm_per_pix = 3.7 / w  # meters per pixel in x dimension
#
# poly_coef = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
# radius = ((1 + (2 * poly_coef[0] * 720 * ym_per_pix + poly_coef[1]) ** 2) ** 1.5) / np.absolute(2 * poly_coef[0])
#
# distance = np.absolute((w // 2 - x[np.max(y)]) * xm_per_pix)


image = cv2.imread("images/main.jpg")

lane_tracker = LaneTracker(image)
overlay_frame = lane_tracker.process(image, draw_lane=True, draw_statistics=True)
cv2.imshow("window", image)
cv2.waitKey(0)
