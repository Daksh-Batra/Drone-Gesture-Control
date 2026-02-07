import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands=mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6)

camera = cv2.VideoCapture(0)
data = "dataset.csv"

if not os.path.exists(data):
    header=[]
    for i in range(21):    #giving csv a header only once as we can take data in many shifts so os path exist is neccessary
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")
    header.append("label")

    with open(data,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(header)

labels={"f":"forward", "b":"backward", "l":"left", "r":"right", "u":"up", "d":"down"}
current_l="forward"
save_count=0

while True:
    success, image = camera.read()
    image=cv2.flip(image, 1)
    color=cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    result=hands.process(color)

    marker_list=None
    if result.multi_hand_landmarks: #if hand detecting
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            marker_list=[]
            for lm in hand_landmarks.landmark:
                marker_list.append(lm.x)
                marker_list.append(lm.y)
                marker_list.append(lm.z)
            #print("landmark count:", len(marker_list))
    
    cv2.putText(image,f"current action: {current_l}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"Saved: {save_count}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Hand Landmark Detection", image)
    
    key=cv2.waitKey(1) & 0xFF
    
    if key == ord("q"): #checks key every 1 millisecond
        break

    if key!=255:
        ch=chr(key)
        if ch in labels:
            current_l=labels[ch]
            print("Current Action:", current_l)
    
    if key==ord("s"):
        if marker_list is not None and current_l is not None:
            row = marker_list + [current_l]

            
            with open(data,"a",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(row)
            
            print("Saved:",current_l)
            save_count+=1

        else:
            print("Not saved(no hand detected or label not set)")