#https://github.com/darshanadakane/yolov3_realTimeObjectDetection/blob/master/YOLO_ObjectDetectionInVideo_git.ipynb
import numpy as np
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description="Object detection with OpenCV")
# Paths 
parser.add_argument('-wt','--model_weights',type=str,default='model_yolov3/yolov3.weights',
                    help= "Path to the pre-trained weights of the network")
parser.add_argument('-cfg','--model_cfg',type=str,default='model_yolov3/yolov3.cfg',
                    help= "Path to the .config file of the network")
parser.add_argument('-cls','--dataset_classes',type=str,default='coco.txt',
                    help= "Path to the classes file of the dataset")
parser.add_argument('-is_realtime','--is_realtime',action='store_false',
                    help="Is real time object detection is being performed")
parser.add_argument('-vid','--vid_path',type=str,default='video.mp4',
                    help= "Path to the video file")

# Parameters
parser.add_argument('-fd','--frame_dimension',type=int,default=320,
                    help= "The dimensions of the frames")
parser.add_argument('-conf_threh','--threshold_confidence',type=int,default=0.3,
                    help= "Threshold confidence for selecting an object")
args = parser.parse_args()
 
net = cv2.dnn.readNet(args.model_weights,args.model_cfg)

classes = []

with open(args.dataset_classes, 'r') as f :
    classes = f.read().splitlines()
print("Classes : ", classes)

# Layers
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))

# Video capturing
print('-----------------------------------------')
if args.is_realtime : 
    print("Real time object detection is being performed")
    cap = cv2.VideoCapture(0)
else :
    print("Video object detection is being performed")
    cap = cv2.VideoCapture(args.video_path)
font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0
starting_time= time.time()

while True:
    _,frame= cap.read() # 
    frame_id+=1
    
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(args.frame_dimension,args.frame_dimension),
        (0,0,0),True,crop=False) #reduce 416 to 320  

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])


    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > args.threshold_confidence:
                #object detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    #print("indexes: ",indexes)


    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(20,50,255),2)
            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
    
    cv2.imshow("Image",frame)
    #key = cv2.waitKey(1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()    
cv2.destroyAllWindows()