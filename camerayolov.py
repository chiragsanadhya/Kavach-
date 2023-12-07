import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import time
# Load COCO labels our YOLO model was trained on
labelspath = "coco.names"
labels = open(labelspath).read().strip().split("\n")

# Initialize the list of colors to represent each possible class label
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Define the path to the YOLO weights and model configuration
weightspath = "yolov3.weights"
configpath = "Unconfirmed 550063.crdownload"

# Load the pre-trained YOLO v3 model
print("[info] loading YOLO from disk")
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

# Open a video capture object
# video_path = "path/to/your/video.mp4"  # Update with your video file path
cap = cv2.VideoCapture(0)

# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# List of object classes for which the box color should be yellow
yellow_classes = ["knife", "scissors", "baseball bat"]

# Loop through frames in the video
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    (h, w) = frame.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutput = net.forward(ln)
    end = time.time()

    print("[info] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classids = []

    for output in layerOutput:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.75:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerx, centery, width, height) = box.astype("int")

                x = int(centerx - (width/2))
                y = int(centery - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classids.append(classId)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [0, 255, 0]  # Default color is green
            text_color = [128, 0, 128] # Purple

            # Change color to yellow for specific classes
            if labels[classids[i]] in yellow_classes:
                color = [0, 255, 255]  # Yellow
                

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(labels[classids[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    cv2.imshow("Video", frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

