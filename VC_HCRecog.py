import cv2 
import time
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

show_im = True
wCam, hCam = 640, 480




devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
asdf = cast(interface, POINTER(IAudioEndpointVolume))

volume_level = asdf.GetMasterVolumeLevel()
volRange = volume = asdf.GetVolumeRange()
minvol = volRange[0]
maxvol = volRange[1]

#hand detection and drawing variables
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils
drawSpec1 = mpDraw.DrawingSpec(thickness = 2, circle_radius = 1)

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}',(30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    #handTracking
    results = hands.process(imgRGB)    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                lineval_1 = None
                lineval_2 = None
                h, w, c,  = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 4:
                    line_val1 = (cx, cy)
                    #print([id, cx, cy], end = '')
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                if id == 8:
                    line_val2 = (cx, cy)
                    
                    #print([id, cx, cy])
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            cv2.line(img, line_val1, line_val2, (190, 190, 190), 2)
            
            lx, ly = int((line_val1[0] + line_val2[0])/2), int((line_val1[1] + line_val2[1])/2)
            cv2.circle(img, (lx, ly), 3, (255, 0, 255), cv2.FILLED)
            
            length_line = math.hypot( line_val2[0] - line_val1[0],line_val2[1] - line_val1[1])
            
            # 9, 150
            # -65, 0
            
            vol_1 = np.interp(length_line, [11, 150], [minvol, maxvol])
            print(vol_1)
            asdf.SetMasterVolumeLevel(vol_1, None)
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, 
                                  drawSpec1, drawSpec1)
    
    
    
    if show_im:
        cv2.imshow("img", img)
        cv2.waitKey(1)
    else:
        print("Video display is not enabled")