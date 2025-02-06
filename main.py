from hand_tracking import HandTracking
from utils import is_within_box
import cv2
from config import BOXES

# Instantiate HandTracking object
hand_tracker = HandTracking()

# Webcam capture
cap = cv2.VideoCapture(0)

# Define boxes for dragging (this can be imported from the config.py file later)
boxes = [{"x": 300, "y": 200, "w": 100, "h": 100, "dragging": False, "prev_x": 0, "prev_y": 0}]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame to detect hands
    index_x, index_y = hand_tracker.detect_hand(frame)

    # Box interaction
    for box in boxes:
        if is_within_box(box, index_x, index_y):
            if not box["dragging"]:
                box["dragging"] = True
                box["prev_x"], box["prev_y"] = index_x, index_y
        elif box["dragging"]:
            box["x"] += index_x - box["prev_x"]
            box["y"] += index_y - box["prev_y"]
            box["prev_x"], box["prev_y"] = index_x, index_y

    # Draw boxes and hands
    for box in boxes:
        cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (255, 0, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking Drag and Drop", frame)

    # 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
