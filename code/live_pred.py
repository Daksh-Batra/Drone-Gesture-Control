import joblib
import cv2
import mediapipe as mp
import numpy as np
import math

model = joblib.load("hand_gesture_model.pkl")
le = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands=mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6)

cam = cv2.VideoCapture(0)

normal_threshold = 0.70
crawl_theshold = 0.59

while True:
    success, image=cam.read()
    image=cv2.flip(image,1)
    color=cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    result=hands.process(color)

    predicted="STOP(NO ACTION)"
    confidence=0.0
    speed_lvl="NONE"
    speed=0.0
    control_mode="NO HAND"
    width=0.0

    if result.multi_hand_landmarks: #if hand detecting
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            lms = hand_landmarks.landmark  #index finger joint 5 and pinky finger joint is 17
            x5,y5=lms[5].x ,lms[5].y
            x17,y17=lms[17].x ,lms[17].y
            width=math.sqrt((x5-x17)**2 + (y5-y17)**2) 

            far=width<0.16
            tracer_list=[]
            for lm in lms:
                tracer_list.append(lm.x)
                tracer_list.append(lm.y)
                tracer_list.append(lm.z)
            X=np.array(tracer_list).reshape(1,-1) #1 row and -1 means as many column as needed
            prob = model.predict_proba(X)[0] #(1,6) as we have 6 classes and we want prob of all and 0 to get the row so its now 1-d
            ind=np.argmax(prob)
            confidence=prob[ind]
            label=le.inverse_transform([ind])[0] #it expects a list or array and 0 to make the output a string eg ["forward"] to "forward"

            if confidence>=normal_threshold:
                control_mode="NORMAL"
                predicted= label.upper()
                if width>0.230:
                    speed=1.0
                    speed_lvl="FAST"
                elif width>0.120:
                    speed=0.7
                    speed_lvl="MEDIUM"
                else:
                    speed=0.45
                    speed_lvl="SLOW"
            elif confidence>=crawl_theshold:
                control_mode="LOW CONFIDENCE"
                predicted= label.upper()
                speed=0.20
                speed_lvl="CRAWL"
            else:
                control_mode="STOP"
                predicted="STOP(NO ACTION)"
                speed=0.0
                speed_lvl="NONE"

    cv2.putText(image,f"Predicted:  {predicted}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Confidence:  {confidence:.2f}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Control Mode:  {control_mode}",(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Palm width:  {width:.3f}",(10,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Speed Level:{speed_lvl} ({speed:.2f})",(10,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


    cv2.imshow("Live Hand Gesture Recognition", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break