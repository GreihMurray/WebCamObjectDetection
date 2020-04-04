#Python program to detect objects in user's webcam and display webcam feed
#Displays webcam feed with objects outlined, named, and the percent confidence

import pandas
import cv2
from imageai.Detection import ObjectDetection
import os

frames = []

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

def findObjects():
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "imgDect.jpg"),
                                                 output_image_path=os.path.join(execution_path, "imagenew.jpg"))


# Assigning our static_back to None
static_back = None

# Initializing DataFrame, one column is start
# time and other column is end time
df = pandas.DataFrame(columns=["Start", "End"])

# Capturing video
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
    # Reading frame(image) from video
    check, frame = video.read()

    # Saves the current frame to the disk  
    cv2.imwrite("imgDect.jpg", frame)

    # Calls the findObject method to run object detection
    findObjects()

    frame = cv2.imread("imagenew.jpg")

    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        break

print("Done")

video.release()

# Destroying all the windows
cv2.destroyAllWindows()
