'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray
import time
import os
import logging as log

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
            if os.path.isfile(self.input_file):
                log.info("Input stream file exists")
            else:
                log.error("Specified input file doesn't exist")
                raise InputFileNotFound
        self.cap_time = []
        
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
            self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_start_time_queue = []
            #print('frame count', self.frame_count)
            #self.cap.open(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            #for _ in range(10):
            cap_start = time.time()
            self.frame_start_time_queue.append(time.time())
            ret, frame=self.cap.read()
            self.pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if ret == 0 :
                break
            self.cap_time.append(1000*(time.time()-cap_start))
            yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

