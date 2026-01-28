import os
import sys
import warnings
import cv2
import numpy as np
import math
import time
import subprocess
import winsound
import pyautogui  # Added for key simulation

# --- PRE-IMPORT CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
pyautogui.FAILSAFE = False

try:
    import mediapipe as mp
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
except (AttributeError, ImportError):
    print("[CRITICAL] MediaPipe failed to load.")
    sys.exit()

class AppSwitcherPro:
    def __init__(self):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            model_complexity=1, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp_draw
        
        self.last_launch_time = 0
        self.cooldown = 3.0  
        self.status_msg = "READY"
        self.msg_expiry = 0
        self.vlc_mode = False # State to track if we are controlling VLC

        self.app_map = {
            1: ["calc", "CALCULATOR"],
            2: [r'"C:\Program Files\VideoLAN\VLC\vlc.exe"', "VLC PLAYER"],
            3: ["start chrome", "GOOGLE CHROME"],
            4: ["start powerpnt", "POWERPOINT"],
            5: ["explorer", "FILE MANAGER"]
        }

    def count_fingers(self, hand_lms):
        fingers = []
        lm = hand_lms.landmark
        # Thumb
        if lm[4].x < lm[3].x: fingers.append(1)
        else: fingers.append(0)
        # 4 Fingers
        tips = [8, 12, 16, 20]
        knuckles = [6, 10, 14, 18]
        for t, k in zip(tips, knuckles):
            if lm[t].y < lm[k].y: fingers.append(1)
            else: fingers.append(0)
        return fingers.count(1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        win_name = "App Switcher HUD"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

        while cap.isOpened():
            success, img = cap.read()
            if not success: break
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            curr_time = time.time()

            cv2.rectangle(img, (0, 0), (w, 80), (40, 40, 40), -1)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    lm = hand_lms.landmark
                    finger_count = self.count_fingers(hand_lms)

                    # --- BRANCH 1: VLC MEDIA CONTROL MODE ---
                    if self.vlc_mode:
                        cv2.putText(img, "VLC REMOTE ACTIVE", (50, 55), 1, 1.5, (0, 165, 255), 3)
                        
                        # 1. PLAY/PAUSE (Pinch Index & Thumb)
                        dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
                        if dist < 0.05:
                            if curr_time - self.last_launch_time > 0.8:
                                pyautogui.press('space')
                                winsound.Beep(600, 50)
                                self.last_launch_time = curr_time
                                self.status_msg = "PLAY / PAUSE"
                                self.msg_expiry = curr_time + 1.0

                        # 2. VOLUME CONTROL (Vertical Index Finger Position)
                        # We use the Y coordinate of the Index Tip
                        vol_level = np.interp(lm[8].y, [0.2, 0.8], [100, 0])
                        if curr_time - self.last_launch_time > 0.1: # Smooth scroll
                            if lm[8].y < 0.4: pyautogui.press('up')
                            elif lm[8].y > 0.6: pyautogui.press('down')

                        # 3. EXIT VLC MODE (Show 5 fingers)
                        if finger_count == 5:
                            self.vlc_mode = False
                            self.status_msg = "EXITING MEDIA MODE"
                            self.msg_expiry = curr_time + 1.5

                    # --- BRANCH 2: STANDARD APP SWITCHER ---
                    else:
                        cv2.putText(img, f"FINGERS: {finger_count}", (50, 55), 1, 1.5, (0, 255, 255), 3)
                        if finger_count in self.app_map:
                            if curr_time - self.last_launch_time > self.cooldown:
                                cmd, name = self.app_map[finger_count]
                                try:
                                    subprocess.Popen(cmd, shell=True)
                                    winsound.Beep(1000, 100)
                                    self.status_msg = f"LAUNCHING: {name}"
                                    if finger_count == 2: self.vlc_mode = True # Enter VLC mode
                                except:
                                    self.status_msg = "ERROR: PATH NOT FOUND"
                                
                                self.last_launch_time = curr_time
                                self.msg_expiry = curr_time + 2.0

            # Display Status Message
            if curr_time < self.msg_expiry:
                cv2.putText(img, self.status_msg, (w//2, 55), 1, 1.2, (0, 255, 0), 3)

            cv2.imshow(win_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AppSwitcherPro()
    app.run()