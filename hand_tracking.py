import cv2
import mediapipe as mp

class HandTracking:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        index_x, index_y = -1, -1

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                index_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return index_x, index_y
