'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import numpy as np
import logging as log
class model:
    '''
    Class for the General Model.
    '''
    def __init__(self, model_name, ie, model_dir, device='CPU',extensions=None, nq = 1, precision = 'FP32'):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model = model_name
        self.device = device
        self.ie = ie
        self.model_dir = model_dir
        self.nq = nq
        self.precision = precision
        self.latency = []
        self.input_processing_time = []
        self.output_processing_time = []

    def load_model(self, model_xml = None):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        if not model_xml:
            model_xml = self.model_dir + "/intel/"+ self.model+"/"+self.precision+"/"+self.model+".xml"
            log.info("Reading IR model from {}".format(model_xml))
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.net = self.ie.read_network(model_xml, model_bin)
        if self.device == 'CPU':
            self.check_layers_support()
        self.exec_net = self.ie.load_network(self.net, self.device, num_requests = self.nq)
        log.info('Loading model ..... OK!')
    def check_layers_support(self):
        
        layers = self.net.layers.keys()
        supported_layers = self.ie.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in layers if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            log.error("Model {} : Following layers are unsupported by {} device: {}".format(self.model, self.device, ','.join(unsupported_layers)))
            raise UnsupportedLayersError
        log.info('Layers support checked ..... OK!')
        return 0
    
    def prepare_input(self, cap):
        self.frame = next(cap)
        prepare_start = time.time()
        self.in_frame = self.preprocess_input()
        self.input_processing_time.append(1000*(time.time() - prepare_start))
        
    
    def async_infer(self, req_ID):
        input_blob = next(iter(self.net.inputs)) #to be changed per class
        Inf_req = self.exec_net.start_async(req_ID, {input_blob:self.in_frame})
    
    def predict(self, cap,mode = 'async'):
        self.mode = mode
        if mode == 'async':
            return self.predict_async(cap)
        elif mode == 'sync':
            return self.predict_sync(cap)
        else:
            log.error('Predict function expects inference mode to be \"sync\" or \"async\" ')
            raise UnknownInferenceMode
    def predict_sync(self, cap):
        log.info('Sync Inference of model {}'.format(self.model))
        while True:
            
            self.prepare_input(cap)
         
            self.async_infer(0)
        
            self.exec_net.requests[0].wait()
            self.latency.append(self.exec_net.requests[0].latency)
            start = time.time()
            outputs = self.get_output(0)
            self.output_processing_time.append(1000*(time.time() - start))
            yield outputs
        
    def predict_async(self, cap):
        log.info('Async Inference of model {}'.format(self.model))
        log.info('Num infer requests = {}'.format(self.nq))
        cur_req_ID = 0
        ready_req_ID = cur_req_ID  - self.nq
        
        self.prepare_input(cap)
        
        start = time.time()
        
        while True:
            if ready_req_ID >= 0:
                self.exec_net.requests[ready_req_ID].wait()
                self.latency.append(self.exec_net.requests[ready_req_ID].latency)
                start = time.time()
                outputs = self.get_output(ready_req_ID)
                self.output_processing_time.append(1000*(time.time() - start))
    
            # infer next frame
            self.async_infer(cur_req_ID)
            
            
            if ready_req_ID >= 0:
                yield outputs #self.frame  # Next execution resumes  
                            # from this point   
            
            cur_req_ID +=1
            ready_req_ID +=1
            if cur_req_ID >= self.nq:
                cur_req_ID = 0
            if ready_req_ID >= self.nq:
                ready_req_ID=0
            # prepare new frame
            #ret, frame = cap.read()
            try:
                self.prepare_input(cap)
                
            except StopIteration:
                break
        for i in range(self.nq):
            self.exec_net.requests[ready_req_ID].wait()
            outputs = self.get_output(ready_req_ID)
            yield outputs
            ready_req_ID +=1
            if ready_req_ID >= self.nq:
                ready_req_ID=0
                    
                
             

    def get_output(self, req_ID):
        outputs = self.exec_net.requests[req_ID].outputs
        key = next(iter(outputs))
        outputs = outputs[key]
        outputs = outputs.squeeze()
        return outputs

    def preprocess_input(self, image = None, input_blob = None):
        if not isinstance(image,np.ndarray): 
            image = self.frame
        if input_blob == None:
            input_blob = next(iter(self.net.inputs))
        if image.any == None:
            log.error("checkpoint ERROR! blank FRAME grabbed")
            raise BlankFrameError
        # Reshaping data
        n, c, h, w = self.net.inputs[input_blob].shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        return in_frame.reshape((n, c, h, w))

class GazeEstimator(model):
    def prepare_input(self, cap):
        LM_eyes, HP_vector = cap
        frame_left_eye, frame_right_eye = next(LM_eyes)
        self.hp_array = next(HP_vector)
        prepare_start = time.time()
        self.left_eye_image = self.preprocess_input(frame_left_eye, 'left_eye_image')
        self.right_eye_image = self.preprocess_input(frame_right_eye,'right_eye_image')
        self.input_processing_time.append(1000*(time.time() - prepare_start))
    
    def async_infer(self, req_ID):
        Inf_req = self.exec_net.start_async(req_ID, {'head_pose_angles': self.hp_array, 'left_eye_image': self.left_eye_image, 'right_eye_image': self.right_eye_image})
    
    def visualize(self, frame, gaze, right_eye, left_eye, color = (255,0,0)):
        dx, dy, _ = gaze
        dx, dy= int(100*dx), int(100*dy)
        thickness = 6
        arrow1_start = right_eye #LandMark.result.right_eye_shifted
        arrow1_end = (arrow1_start[0]+ dx, arrow1_start[1]- dy)
        cv2.arrowedLine(frame, arrow1_start, arrow1_end, 
                                     color, thickness, tipLength=0.3)
        arrow2_start = left_eye #LandMark.result.left_eye_shifted
        arrow2_end = (arrow2_start[0]+ dx, arrow2_start[1]- dy)
        cv2.arrowedLine(frame, arrow2_start, arrow2_end, 
                                     color, thickness, tipLength=0.3)
        
class HeadPoseDetector(model):
    def get_output(self, req_ID):
        outputs = self.exec_net.requests[req_ID].outputs
        key_list = list(outputs)
        array = []
        for i in range(3):
            angle = outputs[key_list[i]].squeeze()
            array.append(int(angle))
        return array
    
    def preprocess_output(self, out_gen):
        while True:
            array = next(out_gen)
            start = time.time()
            pitch , roll , yaw = array
            hp_array = [yaw, pitch, roll]
            self.output_processing_time[-1] +=(1000*(time.time() - start))
            yield hp_array

class LandMarkDetector(model):
    class Result:
        def __init__(self,output, frame):
            self.H, self.W = frame.shape[:2]
            p = []
            for i in range(5):
                p.append((int(output[2*i]*self.W), int(output[2*i+1]*self.H)))
            self.right_eye = p[0]
            self.left_eye = p[1]
        def square_crop_eye(self , p_eye,  l, frame):
            L = int(max(l*self.W, l*self.H)/2)# square side
            xmin, ymin = (p_eye[0]-L, p_eye[1] -L)
            xmax, ymax = (p_eye[0]+L, p_eye[1] +L)
            frame_eye = frame[ymin:ymax, xmin:xmax]
            return frame_eye
        def visualize(self, frame, shift, color = [0,255,0]):
            #print(shift)
            self.left_eye_shifted  = (shift[0]+ self.left_eye[0], shift[1]+ self.left_eye[1]) 
            #print(org)
            cv2.putText(frame,"L",self.left_eye_shifted, cv2.FONT_HERSHEY_SIMPLEX,2, color, 5)
            
            self.right_eye_shifted  = (shift[0]+ self.right_eye[0], shift[1]+ self.right_eye[1])
            cv2.putText(frame,"R",self.right_eye_shifted, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            
    def preprocess_output(self, out_gen):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        while True:
            output = next(out_gen)
            start = time.time()
            self.result = self.Result(output, self.frame)
            # check single face detected
            frame_left_eye = self.result.square_crop_eye(self.result.left_eye,  0.1, self.frame)
            frame_right_eye = self.result.square_crop_eye(self.result.right_eye, 0.1, self.frame)
            self.output_processing_time[-1] +=(1000*(time.time() - start))
            yield frame_left_eye, frame_right_eye

class FaceDetector(model):
    class Result:
        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.top_corner = (output[3], output[4]) # (x, y)
            self.bot_corner = (output[5], output[6]) # (w, h)
        def rescale(self, frame, scale = 1.0):
            H_init, W_init = frame.shape[:-1]
            assert scale >= 1, "scale should be >= 1.0"
            delta = 0.5*(scale -1)* W_init
            x,y = self.top_corner
            self.top_corner = (int(x*W_init -delta), int(y*H_init - delta))
            x,y = self.bot_corner
            self.bot_corner = (int(x*W_init + delta), int(y*H_init + delta))

        def visualize(self, frame, color=[255,0,0]):
            #frameX = cv2.circle(frame, top_corner,10, [255,0,0], -1)
            cv2.rectangle(frame,self.top_corner, self.bot_corner, color, 2)
            return frame
        
        def crop_frame(self, frame):
            xmin, ymin = self.top_corner
            xmax, ymax = self.bot_corner
            frameX = frame[ymin:ymax, xmin:xmax]
            return frameX

    def preprocess_output(self, out_gen, pipeline_branch_count = 2, visualize = False):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        while True:
            outputs = next(out_gen)
            start = time.time()
            self.result = self.Result(outputs[0])
            # check single face detected
            if self.Result(outputs[1]).confidence > self.pt:
                log.error("More than one face is detected with confidence larger than {}".format(self.pt))
                raise MultipleFacesDetected
            self.result.rescale(self.frame,  1.04)
            frame_out = self.result.crop_frame(self.frame)
            
            self.output_processing_time[-1] +=(1000*(time.time() - start))
            for i in range(pipeline_branch_count):
                yield frame_out
    