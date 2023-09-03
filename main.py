import cv2
import time
import json

cap = cv2.VideoCapture(0)
framecount = 0

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

def average(list):
    return sum(list) / len(list)


firsttime = time.time()
with open("info.json",'a') as infodump:
    infodump.write("{}")
data =[]
while True:
    framecount += 1
    ret, frame = cap.read()
    
    # Apply the background subtractor
    fgmask = fgbg.apply(frame)
    

    threshold = 400  
    motion_detected = (cv2.countNonZero(fgmask) > threshold)
    
    re1, frame1 = cap.read()
    framecontours = []
    diff = cv2.absdiff(frame, frame1)
    graydiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(graydiff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not motion_detected:
        no_movement_duration += 1 / 30  
    else:
        no_movement_duration = 0
    if not contours and not motion_detected:
        data = json.load(open('info.json'))
        if type(data) is dict:
            data = [data]

        with open("info.json", 'w') as infodump:
            data.append({framecount: ["No movement", time.time()-firsttime]})
            json.dump(data, infodump, indent=4)  
            infodump.write("\n")
    else:
        with open("info.json",'w') as infodump:
            data.append({framecount:["Movement",time.time()-firsttime]})
            json.dump(data,infodump,indent= 4)
            infodump.write("\n")
    imgdraw = cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('contours', imgdraw)
    
    if (cv2.waitKey(30) == 27):
        break
with open("info.json",'a') as infodump:
    infodump.write("}")
cap.release()
cv2.destroyAllWindows()
