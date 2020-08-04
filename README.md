# Computer Pointer Controller

That is a project that uses Intel OpenVINO toolkit to deploy AI models for PC pointer controller application. 

![Mouse-Control](https://github.com/Aya-ZIbra/PointerController/blob/master/Mouse.JPG?raw=true)

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.


**Models Download**
```bash
cd src
!/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-retail-0004 -o models
!/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o models
!/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o models
!/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o models
```
**pyautogui**
`pip3 install pyautogui`

**Requirments**
All required dependencies and their versions are listed in a requirements.txt file.
 *pipreqs* module was used to clean up my requirements.txt file
Related links:
[bndr/pipreqs: pipreqs - Generate pip requirements.txt file based on imports of any project. Looking for maintainers to move this project forward.](https://github.com/bndr/pipreqs)
[bash - How do I configure PATH variables so I can run packages on the CLI? - Stack Overflow](https://stackoverflow.com/questions/54938229/how-do-i-configure-path-variables-so-i-can-run-packages-on-the-cli)

## Demo
*TODO:* Explain how to run a basic demo of your model.

A basic demo can be run using a bash run script in the src directory.  It takes 4 arguments:
a) Input stream 
b) Device used for Face detection inference
c) Models' precision
d) Number of requests

```bash
cd src
./run.sh ../bin/demo.mp4 CPU FP32 1 
```
The python application has a visualisation flag which is set to False by default. 
The run script sets the visualization to true. If you need to run the script for performance analysis, please remove the "-v".

```bash
INPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
nq=$4

# The default path for the job is the user's home directory,
#  change directory to where the files are.
cd $PBS_O_WORKDIR


python3 demo.py -m models -i $INPUT_FILE \
                            -FP $FP_MODEL \
                            -d $DEVICE\
                            -nq $nq \
                            -v
```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

```
usage: demo.py [-h] -m MODEL_DIR -i INPUT [-d DEVICE] [-l CPU_EXTENSION] [-FP FP] [-nq NUM_REQUESTS] [-c CONFIDENCE]
               [-mouse_precision MOUSE_PRECISION] [-mouse_speed MOUSE_SPEED] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model_dir MODEL_DIR
                        Path to models directory
  -i INPUT, --input INPUT
                        Path to video file or image.'cam' for capturing video stream from camera
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Looksfor a suitable plugin for
                        device specified(CPU by default)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
  -FP FP, --FP FP       Model precision
  -nq NUM_REQUESTS, --num_requests NUM_REQUESTS
                        Number of async requests
  -c CONFIDENCE, --confidence CONFIDENCE
                        Probability threshold for detections filtering
  -mouse_precision MOUSE_PRECISION, --mouse_precision MOUSE_PRECISION
                        mouse controller precision: high, low, medium
  -mouse_speed MOUSE_SPEED, --mouse_speed MOUSE_SPEED
                        mouse controller speed: fast, slow, medium
  -v, --visualize       Visulization of intermediate models. Do not set this flag for performance analysis. This flag forces sync
                        inference mode
```

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

Jobs submitted to an edge node with an [Intel® Core™ i5\-6500TE](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz.html) processor. The inference workload will run on the CPU, GPU or both.
```bash
    qsub run.sh -l nodes=1:i5-6500te -F "../bin/demo.mp4 CPU FP32"
    qsub run.sh -l nodes=1:i5-6500te:intel-hd-530 -F "../bin/demo.mp4 GPU FP32"
```
The table below summarizes the time taken by different operations in the application. Three cases were considered for model precision FP32:
1. All models run synchronously on CPU Core i5
2. Face detection model run *asynchronously* on *CPU*. Other models run synchronously on CPU.
3. Face detection model run *asynchronously* on *GPU*. Other models run synchronously on CPU.

|Operation | Sync CPU | Async CPU | Async CPU (FD on GPU) |
| --- | ---| --- | --- |
|Load time (ms)| 1727.18 | 1588.61 | 22789.98 |
|Frame capture time (ms) | 4.73 | 4.86| 5.17|
| **Model Inference time** | | | |
|Face Detection Latency (ms) | 4.73 | **10.12** ?? | 9.80 |
|LandMark Detection Latency (ms) | 0.34 | 0.35| 0.34 |
|Head pose Detection Latency (ms) | 1.13 |1.22|1.09|
|Gaze Estimation Latency (ms) | 1.01 |1.07|1.18|
| **Input processing** | | | |
|Face Detection Input Processing (ms) | 0.48 | 0.46| 0.52|
|LandMark Detection Input Processing (ms) | 0.05| 0.06| 0.05|
|Head pose Detection Input Processing (ms) | 0.05 | 0.06| 0.05|
|Gaze Estimation Input Processing  (ms) | 0.07 | 0.07| 0.07|
| **Output processing** | | | |
|Face Detection output Processing (ms)| 0.14| 0.21|0.14|
|LandMark Detection output Processing (ms) | 0.09 |0.27|0.10|
|Head pose Detection output Processing (ms) | 0.08 |0.09|0.10|
|Gaze Estimation output Processing  (ms) | 0.05 |0.05|0.08|
| **Throughput (fps)** | 72.00  ( 13.9 ms/frame)| 83.0  ( 12.0 ms/frame)| 92.00 (10.9 ms/frame)|
|**Frame Latency (ms)** | 13.11| 23.54 |20.73|


Here are the performance figures for (Case 3) using FP16:
|Operation | FP32 | FP16 |
| --- | ---| --- | 
|Load time (ms)| 22789.98 | 21924.61|
|Face Detection Latency (ms) | 9.80 | 8.91 |
| **Throughput (fps)** | 92.00 (10.9 ms/frame)| 100.00 ( 10.0 ms/frame)|
|**Frame Latency (ms)** |20.73| 19.24|

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
 
> Notes: 
* The latency of the Face detection model is the most significant so it is worth doing the inference in parallel. In other words, we can use the Async mode for this model. 
* When shifting the FD model to run asynchronously, the latency of the model inferenc increased from 5 ms to 10 ms. Probably, this could be owed to limited Hardware resources. Overall, there is an improvement in the throughput.

* Also, note that the throughput improvement came at the cost of latency increase. This is why we limited the async mode to Face Detection model only. If we did use async mode for all models, we would get a marginal improvement in the throughput while degrading the latency of the frame to 3-4 times the sync mode value.

> For more on the latency vs throughput topic, check the simple Laundry example here: [7.2.1 Latency and Throughput - YouTube](https://www.youtube.com/watch?v=3HIV4MnLGCw)

* Hence, the Face Detection inference was run on an accelerator (GPU). The inference time improved while the loading time degraded.
* Deploying FP16 can slightly decrease the loading time and improve the throughput to 100 fps. 

> For the mouse controller application, real-time fps would be sufficient which is anything above 30 fps. Also, latency is even more important in our case for real-time movement of the mouse. 

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

The performance of async Inference is shown above in the table of the **Benchmarks** section.  
Using the async Inference, the throughput is improved while the latency is slightly increased.

The effects of async inference on power .... 

Effect of using 2 infer requests:
* **CPU** : Latency jumps from about 24 (12x2 ) to 34 (12x3). The throughput is only slightly increased from 82 to 85 fps.
* **GPU** : Latency jumps from about 20 (10x2 ) to 28 (10x3). The throughput is increased from 92 to 102 fps.

**CODE** : A generator function was used to perform async inference for each model:
```python
    def predict(self, cap):
        cur_req_ID = 0
        ready_req_ID = cur_req_ID  - self.nq
        prepare_start = time.time()
        self.prepare_input(cap)
        self.input_processing_time.append(1000*(time.time() - prepare_start))
        start = time.time()
        #print("model start time ", start)
        while True:
            if ready_req_ID >= 0:
                self.exec_net.requests[ready_req_ID].wait()
                self.latency.append(self.exec_net.requests[ready_req_ID].latency)
                outputs = self.get_output(ready_req_ID)
    
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
                prepare_start = time.time()
                self.prepare_input(cap)
                self.input_processing_time.append(1000*(time.time() - prepare_start))
            except StopIteration:
                break
```

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust

One edge case handled by the current code is the multiple faces issue. The code would choose the face with most confidence and makes sure all other detected faces have a cofidence less than a specific threshold value. (-c or --confidence input arg to demo.py). This value is 0.5 by default. 
