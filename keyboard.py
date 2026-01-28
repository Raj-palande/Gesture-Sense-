import os
import sys
import warnings
import cv2
import numpy as np
import math
import time
import pyautogui
import winsound  
import screen_brightness_control as sbc

# --- CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
pyautogui.FAILSAFE = False 

try:
    import mediapipe as mp
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.face_mesh as mp_face
    import mediapipe.python.solutions.drawing_utils as mp_draw
except (AttributeError, ImportError):
    print("[CRITICAL] MediaPipe failed to load.")
    sys.exit()

class VirtualKeyboard:
    def __init__(self):
        # Increased spacing and size for larger window
        self.keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                     ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                     ["Z", "X", "C", "V", "B", "N", "M", ",", ".", " "]]
        self.buttons = []
        # Adjusted coordinates for 1280x720 resolution
        for r, row in enumerate(self.keys):
            for c, key in enumerate(row):
                # Increased button size to 100x100
                self.buttons.append([110 * c + 50, 110 * r + 250, 100, 100, key])

    def draw(self, img, finger_pos=None):
        overlay = img.copy()
        for x, y, w, h, label in self.buttons:
            hover = finger_pos and x < finger_pos[0] < x+w and y < finger_pos[1] < y+h
            color = (0, 255, 0) if hover else (180, 180, 180)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, cv2.FILLED)
            # Increased font size for readability
            cv2.putText(img, label, (x+30, y+65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    def check_click(self, pos):
        for x, y, w, h, label in self.buttons:
            if x < pos[0] < x+w and y < pos[1] < y+h:
                if label == " ":
                    pyautogui.press("space")
                else:
                    pyautogui.press(label.lower())
                winsound.Beep(800, 50) 
                return True
        return False

class GestureSensePro:
    def __init__(self, is_handicapped=False):
        self.is_handicapped = is_handicapped
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.face_mesh = mp_face.FaceMesh(refine_landmarks=True)
        self.keyboard = VirtualKeyboard()
        self.show_kb = True 
        self.plocX, self.plocY = 0, 0
        self.smoothening = 5
        self.last_action_time = 0
        self.screen_w, self.screen_h = pyautogui.size()

    def run(self):
        window_name = "GestureSense Pro Preview"
        # Changed to WINDOW_AUTOSIZE to respect the set resolution
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        cap = cv2.VideoCapture(0)
        # --- INCREASED RESOLUTION ---
        cap.set(3, 1280) # Width
        cap.set(4, 720)  # Height

        while cap.isOpened():
            success, img = cap.read()
            if not success: break
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.is_handicapped:
                results = self.face_mesh.process(img_rgb)
                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark
                    nose = lms[1]
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    # Mapping nose to screen coordinates
                    mx = np.interp(nx, [w*0.3, w*0.7], [0, self.screen_w])
                    my = np.interp(ny, [h*0.3, h*0.7], [0, self.screen_h])
                    self.plocX = self.plocX + (mx - self.plocX) / self.smoothening
                    self.plocY = self.plocY + (my - self.plocY) / self.smoothening
                    pyautogui.moveTo(self.plocX, self.plocY)
                    
                    if self.show_kb:
                        img = self.keyboard.draw(img, (nx, ny))
                        if abs(lms[159].y - lms[145].y) < 0.007:
                            if time.time() - self.last_action_time > 0.7:
                                self.keyboard.check_click((nx, ny))
                                self.last_action_time = time.time()
            else:
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        lm = hand_lms.landmark
                        # Index tip landmark
                        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                        dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
                        
                        img = self.keyboard.draw(img, (ix, iy))
                        
                        if dist < 0.06 and (time.time() - self.last_action_time > 0.5):
                            self.keyboard.check_click((ix, iy))
                            self.last_action_time = time.time()
                            cv2.circle(img, (ix, iy), 25, (0, 255, 0), cv2.FILLED)
                        else:
                            cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)

            cv2.putText(img, "HD PREVIEW ACTIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = input("Handicapped Mode? (yes/no): ").lower().strip()
    app = GestureSensePro(is_handicapped=(choice == 'yes'))
    app.run()
    