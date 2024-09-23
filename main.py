import cv2
import numpy as np
import mediapipe as mp
import os
import time
import threading
import tkinter as tk
from PIL import Image
import customtkinter as ctk
import webbrowser
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class HandGestureDetector:
    """A class that handles hand gesture detection using MediaPipe and triggers specific actions."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cooldown_time = 5  # Cooldown for triggering gestures (in seconds)
        self.last_trigger_time = 0  # Time when the last action was triggered

    def is_finger_up(self, hand_landmarks, finger_tip_idx, finger_dip_idx):
        """Helper function to check if a specific finger is raised."""
        return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_dip_idx].y

    def detect_gesture(self, hand_landmarks):
        """Detects the current hand gesture based on finger positions."""
        tips = [self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP]
        
        dips = [self.mp_hands.HandLandmark.THUMB_IP, self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP, self.mp_hands.HandLandmark.RING_FINGER_DIP,
                self.mp_hands.HandLandmark.PINKY_DIP]
        
        fingers_up = [self.is_finger_up(hand_landmarks, tip.value, dip.value) for tip, dip in zip(tips, dips)]

        if fingers_up[1] and fingers_up[2] and not any(fingers_up[3:]):
            return "two_fingers"
        elif all(fingers_up):
            return "open_palm"
        elif fingers_up[1] and not any(fingers_up[2:]):
            return "pointing"
        elif not any(fingers_up):
            return "fist"
        else:
            return "unknown"

    def trigger_action(self, gesture):
        """Triggers an action based on the detected gesture."""
        current_time = time.time()

        # Only trigger actions if cooldown time has passed
        if current_time - self.last_trigger_time > self.cooldown_time:
            if gesture == "two_fingers":
                # print("Opening YouTube...")
                webbrowser.open("https://www.youtube.com")
            elif gesture == "open_palm":
                # print("Opening Notepad...")
                os.system("notepad" if os.name == 'nt' else "gedit &")
            elif gesture == "pointing":
                # print("Opening Calculator...")
                os.system("calc" if os.name == 'nt' else "gnome-calculator &")
            elif gesture == "fist":
                # print("Opening Terminal...")
                os.system("start cmd" if os.name == 'nt' else "gnome-terminal &")
            
            self.last_trigger_time = current_time  # Update the last trigger time


class HandGestureApp:
    """Main application class that handles the GUI and camera feed."""
    
    def __init__(self):
        # Initialize the GUI
        self.app = ctk.CTk()
        self.app.geometry("1200x700")
        self.app.title("Hand Command Pro")
        # Center the window
        self.center_window(1000, 550)

        # Initialize UI elements
        self.camera_label = ctk.CTkLabel(self.app, text="")
        self.camera_label.pack(pady=20)

        self.open_button = ctk.CTkButton(master=self.app, text="Open Camera", command=self.start_camera)
        self.open_button.pack(pady=20)

        self.stop_button = ctk.CTkButton(master=self.app, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack(pady=20)

        # Initialize Hand Gesture Detector
        self.gesture_detector = HandGestureDetector()

        # Camera variables
        self.cap = None
        self.camera_running = False
        self.frame = None

    def center_window(self, width, height):
        # Get screen width and height
        screen_width = self.app.winfo_screenwidth()
        screen_height = self.app.winfo_screenheight()

        # Calculate position x and y coordinates
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Set window geometry
        self.app.geometry(f'{width}x{height}+{x}+{y}')

    def start_camera(self):
        """Starts the camera feed and gesture detection in a separate thread."""
        if not self.camera_running:
            self.camera_running = True
            self.cap = cv2.VideoCapture(0)
            
            # Start camera feed in a separate thread
            threading.Thread(target=self.camera_feed, daemon=True).start()

            # Update GUI with the camera feed
            self.update_gui()

    def stop_camera(self):
        """Stops the camera feed."""
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()

    def camera_feed(self):
        """Handles capturing frames from the camera and detecting gestures."""
        while self.camera_running:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # Flip and process the frame for hand gestures
            self.frame = cv2.flip(self.frame, 1)
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            result = self.gesture_detector.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.gesture_detector.mp_drawing.draw_landmarks(
                        self.frame, hand_landmarks, self.gesture_detector.mp_hands.HAND_CONNECTIONS)

                    # Detect gesture and trigger actions
                    gesture = self.gesture_detector.detect_gesture(hand_landmarks)
                    if gesture != "unknown":
                        self.gesture_detector.trigger_action(gesture)

            time.sleep(0.03)  # Sleep to simulate frame delay (30 FPS)

    def update_gui(self):
        """Updates the camera feed in the GUI."""
        if self.camera_running and self.frame is not None:
            # Convert the frame to a format Tkinter can use
            img = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, size=(640, 480))

            # Update the camera label with the latest image
            self.camera_label.configure(image=imgtk)
            self.camera_label.image = imgtk

        # Schedule the next GUI update after 10ms
        if self.camera_running:
            self.camera_label.after(10, self.update_gui)

    def run(self):
        """Runs the main application loop."""
        self.app.mainloop()


if __name__ == "__main__":
    app = HandGestureApp()
    app.run()
