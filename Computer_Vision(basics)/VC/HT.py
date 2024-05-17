import cv2
import mediapipe as mp
import time

def rescaleFrame(frame,scale=0.75):     # by 25% down scaled
    width=int(frame.shape[1]*scale)     # shape[1] is width  and then typecasted
    height=int(frame.shape[0]*scale)    # shape[0] is height and then typecasted
    dimensions=(width,height)

    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)

#cv2.waitKey(0)
#ip = "192.168.176.36"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# to measure the FPS
prev_time = 0
pres_time = 0

cap = cv2.VideoCapture(1)
while True:
    success, img = cap.read()
    rescaled_ver=rescaleFrame(img)
    img_RGB = cv2.cvtColor(rescaled_ver, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rescaled_ver) #process the frame
    #checking for multiple hands detected or not 
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # to chck the id number of each hands
            for id, lm in enumerate(hand_landmark.landmark):
                # print(id, lm)
                h, w, c = rescaled_ver.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # making the first id point of the hand bigger i.e id number 0
                # iqcf id == 0:
                #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                cv2.circle(rescaled_ver, (cx, cy), 7, (255, 0, 255), cv2.FILLED)


            mp_draw.draw_landmarks(rescaled_ver, hand_landmark, mp_hands.HAND_CONNECTIONS)

    
    # measuring FPS
    pres_time = time.time()
    fps = 1/(pres_time - prev_time)
    prev_time = pres_time
    cv2.putText(img, f"FPS: {str(int(fps))}", (30, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0 , 0), 2)

    cv2.imshow("Hand Tracking", rescaled_ver)
    if cv2.waitKey(1) == ord('d'):
        break
    
cap.release()
cv2.destroyAllWindows()