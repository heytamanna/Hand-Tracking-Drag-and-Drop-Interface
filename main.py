import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

#if you  want to add more boxes you can add here
boxes = [
    {"x": 300, "y": 200, "w": 100, "h": 100, "dragging": False, "prev_x": 0, "prev_y": 0},
    #{"x": 500, "y": 200, "w": 100, "h": 100, "dragging": False, "prev_x": 0, "prev_y": 0},
]

# Check if the hand is within the bounds of a box
def is_within_box(box, x, y):
    return (box["x"] < x < box["x"] + box["w"]) and (box["y"] < y < box["y"] + box["h"])

#webcam start capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    results = hands.process(frame_rgb) 

    #default values for index_x and index_y
    index_x, index_y = -1, -1

    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks:
            index_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            index_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
            
            for box in boxes:
                if is_within_box(box, index_x, index_y):
                    if not box["dragging"]:  
                        box["dragging"] = True
                        box["prev_x"], box["prev_y"] = index_x, index_y
                elif box["dragging"]:
                    
                    box["x"] += index_x - box["prev_x"]
                    box["y"] += index_y - box["prev_y"]
                    box["prev_x"], box["prev_y"] = index_x, index_y

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for box in boxes:
        cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (255, 0, 0), -1)

    
    if index_x == -1 or index_y == -1:
        for box in boxes:
            box["dragging"] = False

    # Display the frame
    cv2.imshow("Hand Tracking Drag and Drop", frame)

    # 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
