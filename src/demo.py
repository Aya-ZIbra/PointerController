from openvino.inference_engine import IECore
from model import *
import os
import cv2
import time
from argparse import ArgumentParser
from input_feeder import InputFeeder
#from mouse_controller import MouseController

### arguments ## ADD ARGPARSER
def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_dir", required=True,
                        help="Path to models directory")
    
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                        "'cam' for capturing video stream from camera")
    
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-FP", "--FP", default='FP32', type=str,
                        help="Model precision")
    
    parser.add_argument("-nq", "--num_requests", default =1, type=int, help= "Number of async requests")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument('-o', "--output_file", default = None, type=str, help = "Demo output video. Created only if -o is given")
    
    return parser

def main():
    args = args_parser().parse_args()
    model_dir = args.model_dir
    FP = args.FP
    device = args.device
    input_stream = args.input
    out_video = args.output_file              
    FD_model = "face-detection-retail-0004"
    LM_model = 'landmarks-regression-retail-0009'
    HP_model = "head-pose-estimation-adas-0001"
    GE_model = "gaze-estimation-adas-0002"
    sync = 1
    ####

    ie = IECore()
    #Loading models
    load_start = time.time()
    FaceDetect = FaceDetector(FD_model, ie, model_dir, device, nq = args.num_requests , precision = FP )
    FaceDetect.load_model()
    FaceDetect.pt = args.confidence
    LandMark = LandMarkDetector(LM_model, ie, model_dir, 'CPU', precision = FP )
    LandMark.load_model()
    HeadPose = HeadPoseDetector(HP_model, ie, model_dir, 'CPU', precision = FP )
    HeadPose.load_model()
    GE = GazeEstimator(GE_model, ie, model_dir, 'CPU', precision = FP )
    GE.load_model()
    
    load_time = 1000*(time.time()- load_start)
    with open( 'stats_'+FP+'_'+device+'.txt', 'w') as f:
        f.write("Load time = {:.2f} ms \n".format(load_time)) 

    if input_stream == 'cam':
        feed=InputFeeder(input_type='cam')
    else:
        feed=InputFeeder(input_type='video', input_file= input_stream)
    feed.load_data()
    feed_gen = feed.next_batch()


    FD_outgen = FaceDetect.predict(feed_gen, 'async')
    FD_facegen = FaceDetect.preprocess_output(FD_outgen,2)
    

    LM_outgen = LandMark.predict(FD_facegen, 'sync')
    LM_eyes = LandMark.preprocess_output(LM_outgen)

    HP_outgen = HeadPose.predict(FD_facegen, 'sync')
    HP_vector = HeadPose.preprocess_output(HP_outgen)

    gaze_outgen = GE.predict(( LM_eyes, HP_vector), 'sync')

    ### Init Mouse controller #####################
    #mouse = MouseController('medium', 'medium')
    ### Init Video Demo ###########################
    if out_video:
        cap = cv2.VideoCapture("/home/u37265/Project3-Udacity/PointerController/bin/demo.mp4")
        cap.open("/home/u37265/Project3-Udacity/PointerController/bin/demo.mp4")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_point = (W//4, H//4)
        
        cap.release()

        mousevid = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H), True)
    ################################################
    start = time.time()
    i = 0
    latency = []
    for gaze in gaze_outgen:
        dx , dy, dz = gaze
        latency.append(1000*(time.time()-feed.frame_start_time_queue.pop(0)))
            
        frame = FaceDetect.frame
        #### Move mouse
        #mouse.move(dx,dy)
        #### Create Vidoe Demo 
        if out_video:
            dx, dy = int(10*dx) , int(10*dy)
            end_point = (start_point[0]+ dx, start_point[1]- dy)
            #time_stamp1 = time.time()
            framex = cv2.circle(frame, end_point ,20, [255,0,0], -1)
            mousevid.write(framex)
            #wframe = 1000*(time.time()-time_stamp1)
            #print(wframe, " ms") #average 10 ms
            start_point = end_point
        i+=1
    print("i = ", i, "and feed.pos = ", feed.pos)
    tp = feed.frame_count//(time.time()-start)
    with open( 'stats_'+FP+'_'+device+'.txt', 'a') as f:
        f.write("## Frame capture time (ms) = {:.2f} \n".format(np.mean(feed.cap_time)))
        
        f.write("## Model Inference time \n")
        f.write("Face Detection Latency (ms) = {:.2f} \n".format(np.mean(FaceDetect.latency)))
        
        f.write("LandMark Detection Latency (ms) = {:.2f} \n".format(np.mean(LandMark.latency)))
        
        f.write("Head pose Detection Latency (ms) = {:.2f} \n".format(np.mean(HeadPose.latency)))
        
        f.write("Gaze Estimation Latency (ms) = {:.2f} \n".format(np.mean(GE.latency)))
        
        if sync:
            f.write("## Input processing \n")
            f.write("Face Detection Input Processing (ms) = {:.2f} \n".format(np.mean(FaceDetect.input_processing_time)))
            f.write("LandMark Detection Input Processing (ms) = {:.2f} \n".format(np.mean(LandMark.input_processing_time)))
            f.write("Head pose Detection Input Processing (ms) = {:.2f} \n".format(np.mean(HeadPose.input_processing_time)))
            f.write("Gaze Estimation Input Processing  (ms) = {:.2f} \n".format(np.mean(GE.input_processing_time)))
        
        f.write("## Output processing \n")
        f.write("Face Detection output Processing (ms) = {:.2f} \n".format(np.mean(FaceDetect.output_processing_time)))
        f.write("LandMark Detection output Processing (ms) = {:.2f} \n".format(np.mean(LandMark.output_processing_time)))
        f.write("Head pose Detection output Processing (ms) = {:.2f} \n".format(np.mean(HeadPose.output_processing_time)))
        f.write("Gaze Estimation output Processing  (ms) = {:.2f} \n".format(np.mean(GE.output_processing_time)))
        
        f.write("\n throughput = {:.2f} fps ( {} ms per frame) \n".format(tp , 1000/tp ))  
        f.write("Frame latency (ms):  {} \n".format(latency[:10] ) ) 
        f.write("Average Frame Latency (ms): {:0.2f} \n".format(np.mean(latency[10:-10])))
if __name__ == '__main__':
    main()