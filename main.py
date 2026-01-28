import os
import sys
import warnings
import cv2
import numpy as np
import math
import time
import pyautogui
import webbrowser
import screen_brightness_control as sbc

# --- PRE-IMPORT CONFIGURATION ---
# Silence TensorFlow and oneDNN noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)

# Prevent PyAutoGUI from crashing if mouse hits a corner (Critical for Face Mode)
pyautogui.FAILSAFE = False 

try:
    import mediapipe as mp
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.face_mesh as mp_face
    import mediapipe.python.solutions.drawing_utils as mp_draw
except (AttributeError, ImportError):
    print("[CRITICAL] MediaPipe failed to load. Ensure you are on Python 3.12 with compatible NumPy.")
    sys.exit()

class GestureSensePro:
    def __init__(self, is_handicapped=False):
        self.is_handicapped = is_handicapped
        
        # Hand Detection Setup
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            model_complexity=1, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        
        # Face Detection Setup (Accessibility)
        self.mp_face = mp_face
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        
        self.mp_draw = mp_draw
        
        # System State Variables
        self.plocX, self.plocY = 0, 0
        self.smoothening = 5
        self.last_action_time = 0
        self.gesture_cooldown = 1.0
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.dragging = False
        self.frame_reduction = 100 

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        mode_text = "MODE: ACCESSIBILITY (FACE)" if self.is_handicapped else "MODE: STANDARD (HAND)"
        print(f"[INFO] System started in {mode_text} Mode.")

        while cap.isOpened():
            success, img = cap.read()
            if not success: break
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- BRANCH 1: ACCESSIBILITY MODE (FACE CONTROL) ---
            if self.is_handicapped:
                results = self.face_mesh.process(img_rgb)
                if results.multi_face_landmarks:
                    face_lms = results.multi_face_landmarks[0].landmark
                    
                    # 1. NOSE TIP (Landmark 1) for Cursor Movement
                    nose = face_lms[1]
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    
                    # Coordinate Mapping: Translates nose movement to screen coordinates
                    mx = np.interp(nx, [250, 390], [0, self.screen_w])
                    my = np.interp(ny, [200, 280], [0, self.screen_h])
                    
                    # Apply smoothening to prevent cursor jitter
                    clocX = self.plocX + (mx - self.plocX) / self.smoothening
                    clocY = self.plocY + (my - self.plocY) / self.smoothening
                    pyautogui.moveTo(clocX, clocY)
                    self.plocX, self.plocY = clocX, clocY

                    # 2. LEFT EYE WINK (Landmarks 159, 145) for Mouse Click
                    # Measures vertical distance between eyelids
                    eye_dist = abs(face_lms[159].y - face_lms[145].y)
                    if eye_dist < 0.007: 
                        pyautogui.click()
                        cv2.putText(img, "EVENT: LEFT CLICK", (50, 80), 1, 1.5, (0, 255, 0), 2)
                    
                    # 3. JAW DROP (Landmarks 13, 14) for ENTER KEY
                    mouth_dist = abs(face_lms[13].y - face_lms[14].y)
                    if mouth_dist > 0.05:
                        curr_t = time.time()
                        if curr_t - self.last_action_time > 1.5:
                            pyautogui.press('enter')
                            cv2.putText(img, "EVENT: ENTER PRESSED", (50, 120), 1, 1.5, (0, 255, 255), 2)
                            self.last_action_time = curr_t

                    # 4. HEAD PITCH (Landmark 10 vs 1) for SCROLLING
                    tilt = face_lms[10].y - nose.y
                    if tilt > -0.04: # Tilted Down
                        pyautogui.scroll(-25)
                        cv2.putText(img, "SCROLL: DOWN", (w-200, 30), 1, 1, (255, 0, 255), 1)
                    elif tilt < -0.08: # Tilted Up
                        pyautogui.scroll(25)
                        cv2.putText(img, "SCROLL: UP", (w-200, 30), 1, 1, (255, 0, 255), 1)
                    
                    cv2.circle(img, (nx, ny), 5, (255, 0, 255), cv2.FILLED)

            # --- BRANCH 2: STANDARD MODE (HAND GESTURES) ---
            else:
                cv2.rectangle(img, (self.frame_reduction, self.frame_reduction), 
                              (w - self.frame_reduction, h - self.frame_reduction), (255, 0, 255), 2)
                
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                        lm = hand_lms.landmark

                        # Check Finger States
                        index_up = lm[8].y < lm[6].y
                        middle_up = lm[12].y < lm[10].y
                        ring_up = lm[16].y < lm[14].y
                        pinky_up = lm[20].y < lm[18].y
                        thumb_up = math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) > 0.06
                        curr_t = time.time()

                        # GESTURE: BRIGHTNESS (Thumb + Pinky)
                        if thumb_up and pinky_up and not index_up and not middle_up:
                            angle = math.degrees(math.atan2(lm[4].y - lm[20].y, lm[4].x - lm[20].x))
                            bright = np.interp(angle, [-150, 150], [0, 100])
                            sbc.set_brightness(int(bright))
                            cv2.putText(img, f"BRIGHT: {int(bright)}%", (10, 80), 1, 1.5, (255, 255, 0), 2)

                        # GESTURE: SLIDESHOW NEXT (Open Palm)
                        elif index_up and middle_up and ring_up and pinky_up:
                            if curr_t - self.last_action_time > self.gesture_cooldown:
                                pyautogui.press('right')
                                cv2.putText(img, "EVENT: NEXT", (200, 70), 1, 2, (0, 255, 0), 2)
                                self.last_action_time = curr_t

                        # GESTURE: SLIDESHOW PREV (Thumb Only)
                        elif thumb_up and not index_up and not middle_up and not pinky_up:
                            if curr_t - self.last_action_time > self.gesture_cooldown:
                                pyautogui.press('left')
                                cv2.putText(img, "EVENT: PREV", (200, 70), 1, 2, (0, 0, 255), 2)
                                self.last_action_time = curr_t

                        # GESTURE: SCROLLING (Index + Middle)
                        elif index_up and middle_up and not ring_up and not pinky_up:
                            scroll_val = np.interp(lm[8].y, [0.2, 0.8], [30, -30])
                            pyautogui.scroll(int(scroll_val))

                        # GESTURE: MOUSE MOVE & DRAG (Index Only)
                        elif index_up and not middle_up and not pinky_up:
                            tx, ty = int(lm[8].x * w), int(lm[8].y * h)
                            mx = np.interp(tx, [self.frame_reduction, w - self.frame_reduction], [0, self.screen_w])
                            my = np.interp(ty, [self.frame_reduction, h - self.frame_reduction], [0, self.screen_h])
                            
                            clocX = self.plocX + (mx - self.plocX) / self.smoothening
                            clocY = self.plocY + (my - self.plocY) / self.smoothening
                            pyautogui.moveTo(clocX, clocY)
                            self.plocX, self.plocY = clocX, clocY

                            # Drag & Drop via Pinch
                            if math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y) < 0.04:
                                if not self.dragging:
                                    pyautogui.mouseDown()
                                    self.dragging = True
                                cv2.putText(img, "STATUS: DRAGGING", (10, 110), 1, 1, (0, 0, 255), 2)
                            else:
                                if self.dragging:
                                    pyautogui.mouseUp()
                                    self.dragging = False

            # Display Status HUD
            cv2.putText(img, mode_text, (10, 35), 1, 1.3, (0, 255, 255), 2)
            cv2.imshow("GestureSense Pro Final Year Project", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("      GESTURE SENSE PRO - UNIVERSAL HCI      ")
    print("      (Accessibility & Standard Control)      ")
    print("="*50)
    
    choice = input("\nEnable Accessibility Mode for Handicapped Users? (yes/no): ").lower().strip()
    
    app = GestureSensePro(is_handicapped=(choice == 'yes'))
    app.run()