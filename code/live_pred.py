import joblib
import cv2
import mediapipe as mp
import numpy as np
import math
import time

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
armed = False

palm_st=None
fist_st=None
peace_st=None

hold_time=4.0

def rock(lms):
    index_cl=lms[8].y < lms[6].y #8 is tip and 6 is joint(pip) of index finger
    middle_cl=lms[12].y > lms[10].y 
    ring_cl=lms[16].y > lms[14].y 
    pinky_cl=lms[20].y < lms[18].y

    return index_cl and middle_cl and ring_cl and pinky_cl

def palm(lms):
    index_op=lms[8].y < lms[6].y #8 is tip and 6 is joint(pip) of index finger
    middle_op=lms[12].y < lms[10].y 
    ring_op=lms[16].y < lms[14].y 
    pinky_op=lms[20].y < lms[18].y

    return index_op and middle_op and ring_op and pinky_op

def peace(lms):
    index_op=lms[8].y < lms[6].y #8 is tip and 6 is joint(pip) of index finger
    middle_op=lms[12].y < lms[10].y 
    ring_op=lms[16].y > lms[14].y 
    pinky_op=lms[20].y > lms[18].y

    return index_op and middle_op and ring_op and pinky_op

zoom = 1.0
zoom_step = 0.15
min_zoom = 1.0
max_zoom = 5.0

pr_pinch=None
pinch_lock=False
zoom_threshold=0.03
unlock_threshold=0.005

drone_motion=True

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

            palm_detected = palm(lms)
            fist_detected = rock(lms)
            
            x5,y5=lms[5].x ,lms[5].y
            x17,y17=lms[17].x ,lms[17].y
            width=math.sqrt((x5-x17)**2 + (y5-y17)**2) 

            current_tm=time.time()
            if palm_detected:
                if palm_st is None:
                    palm_st = current_tm
                fist_st = None
                if (current_tm - palm_st) >= hold_time:
                    armed = False
                    palm_st = None
            elif fist_detected:
                if fist_st is None:
                    fist_st = current_tm
                palm_st = None
                if  (current_tm - fist_st) >= hold_time:
                    armed = True
                    fist_st = None
            else:
                palm_st = None
                fist_st = None
            
            if armed:
                if peace(lms):
                    if peace_st is None:
                        peace_st = current_tm
                    if (current_tm - peace_st) >= hold_time:
                        drone_motion= not drone_motion
                        peace_st=None
                if drone_motion:
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
                    
                    pr_pinch=None
                    pinch_lock=False

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
                else:
                    control_mode="DRONE CAMERA MODE"
                    predicted="STOP(NO ACTION)"
                    confidence=0.0
                    speed=0.0
                    speed_lvl="NONE"
                            
                    thmb=lms[4]
                    index=lms[8]
                    pinch_dist=math.sqrt((thmb.x - index.x)**2 + (thmb.y - index.y)**2)
                    if pr_pinch is None:
                        pr_pinch=pinch_dist
                    diff=pinch_dist-pr_pinch # >0 then zoom in else zoom out

                    if not pinch_lock:
                        if abs(diff)>zoom_threshold:
                            zoom+=zoom_step if diff>0 else -zoom_step
                            pinch_lock=True
                    if abs(diff)<unlock_threshold:
                        pinch_lock=False

                    zoom=max(min_zoom,min(zoom,max_zoom))
                    pr_pinch=pinch_dist

            else:
                control_mode="DISARMED"
                predicted="STOP(NO ACTION)"
                confidence=0.0
                speed=0.0
                speed_lvl="NONE"
            

    safety = "DISENGAGED" if armed else "ENGAGED"
    cv2.putText(image,f"Predicted:  {predicted}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Confidence:  {confidence*100:.2f}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Control Mode:  {control_mode}",(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Palm width:  {width:.3f}",(10,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Speed Level:{speed_lvl} ({speed:.2f})",(10,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Safety:{safety}",(10,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Zoom:{zoom:.2f}x",(10,280),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Live Hand Gesture Recognition", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break   