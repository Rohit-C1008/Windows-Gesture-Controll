import cv2
import mediapipe as mp
import time

#initializing the webcam
cap = cv2.VideoCapture(0)
wCam, hCam = 1200, 900
cap.set(3, wCam)
cap.set(4, hCam)

#time
pTime = 0
cTime = 0

#hand_detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence = 0.5)
#face_detection
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()



#Drawing tool
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 2, circle_radius = 2)
drawSpec1 = mpDraw.DrawingSpec(thickness = 2, circle_radius = 2)

#Setting up the webcam display
while True:
    
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Hands
    results = hands.process(imgRGB)    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c,  = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
    
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
           
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, 
                                  drawSpec1, drawSpec1)
           
    #Face
    results1 = faceMesh.process(imgRGB)
    if results1.multi_face_landmarks:
        for faceLms in results1.multi_face_landmarks:
           mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS #FACEMESH_TESSELATIONS
                                 , drawSpec, drawSpec)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img
                ,str(int(fps))
                ,(10,70)
                ,cv2.FONT_HERSHEY_PLAIN
                ,3
                ,(255, 255, 255)
                ,2
                )                   
    cv2.flip(img, 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)