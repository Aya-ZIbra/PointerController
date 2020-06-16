from openvino.inference_engine import IECore
from model import *
import os
import cv2
import time
from input_feeder import InputFeeder
#from mouse_controller import MouseController

### arguments ## ADD ARGPARSER
model_dir = "/home/u37265/Project3-Udacity/PointerController/src/models"
FD_model = "face-detection-retail-0004"
LM_model = 'landmarks-regression-retail-0009'
HP_model = "head-pose-estimation-adas-0001"
GE_model = "gaze-estimation-adas-0002"
video = '/home/u37265/Project3-Udacity/PointerController/bin/demo.mp4'
####

ie = IECore()
FaceDetector = FaceDetector(FD_model, ie, model_dir)
FaceDetector.load_model()
LandMark = LandMarkDetector(LM_model, ie, model_dir)
LandMark.load_model()
HeadPose = HeadPoseDetector(HP_model, ie, model_dir)
HeadPose.load_model()
GE = GazeEstimator(GE_model, ie, model_dir)
GE.load_model()


feed=InputFeeder(input_type='video', input_file= video)
feed.load_data()
feed_gen = feed.next_batch()


FD_outgen = FaceDetector.predict(feed_gen)
FD_facegen = FaceDetector.preprocess_output(FD_outgen,2)
    

LM_outgen = LandMark.predict(FD_facegen)
LM_eyes = LandMark.preprocess_output(LM_outgen)

HP_outgen = HeadPose.predict(FD_facegen)
HP_vector = HeadPose.preprocess_output(HP_outgen)

gaze_outgen = GE.predict(( LM_eyes, HP_vector))

### Init Mouse controller #####################
#mouse = MouseController('medium', 'medium')
### Init Video Demo ###########################
cap = cv2.VideoCapture("/home/u37265/Project3-Udacity/PointerController/bin/demo.mp4")
cap.open("/home/u37265/Project3-Udacity/PointerController/bin/demo.mp4")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
start_point = (W//4, H//4)

mousevid = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H), True)
################################################
start = time.time()
i = 0

for gaze in gaze_outgen:
    dx , dy, dz = gaze
    frame = FaceDetector.frame
    #### Move mouse
    #mouse.move(dx,dy)
    #### Create Vidoe Demo 
    dx, dy = int(10*dx) , int(10*dy)
    end_point = (start_point[0]+ dx, start_point[1]- dy)
    #time_stamp1 = time.time()
    #framex = cv2.circle(frame, end_point ,20, [255,0,0], -1)
    #mousevid.write(framex)
    #wframe = 1000*(time.time()-time_stamp1)
    #print(wframe, " ms") #average 10 ms
    start_point = end_point
    i+=1

tp = feed.frame_count//(time.time()-start)
print("throughput = {} fps ( {} ms per frame)".format(tp , 1000/tp ))    
print("i = ", i, "and feed.pos = ", feed.pos)

#cap.release()