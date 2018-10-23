import numpy as np
import cv2
from PIL import ImageGrab

import dlib
# img = ImageGrab.grab(bbox=(10,10,500,500))
# img_np = np.array(img)
# frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
# cv2.imshow("frame", frame)
# cv2.imshow("frame", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
win = dlib.image_window()

def screen_record():
    while(True):
        image = np.array(ImageGrab.grab())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        print("capturing")
        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

screen_record()
win.set_image(ImageGrab.grab())

# When everything done, release the capture
cv2.destroyAllWindows()