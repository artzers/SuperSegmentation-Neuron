# SuperSegmentation-Neuron

### 1.Requirement
Our Training and Test environment is Python3.7.3, Pytorch 1.5.1+Cuda 9.2. 
Other python packages:
- Numpy
- tqdm
- tifffile
- visdom
- scanf

### 2.Prepare
(1) Use neurite tracing tools such like NeuroGPS-Tree (https://github.com/GTreeSoftware/GTree), Vaa3d(https://github.com/Vaa3D) to trace the neurites. The traced neurites are saved in SWC files. The SWC files from the same neuron should be saved in one directory. Use the image name to name the corresponding directory. Each original image is also put into the corresponding directory.

(2) Edit MergeSwc.py. Change the parameter “srcPath” to the root path of SWC directories. Then run MergeSwc.py. All of the SWC files in the same directory will be merged as one SWC file, which is named as “XXX_allSwc.swc”.

(3) Edit the SampleMaker.py. Change the parameter “srcPath” to the root path of SWC directories. Run SampleMaker.py. These python codes will read the original image and  “XXX_allSwc.swc” in each directory, and then generate the segmented image of high resolution as the ground truth in the current directory.


### 3.Training
(1) Build new directories named “original” and “bin”. 

(2) Put all of the original images into "original" directory and ground truth images into "bin" directory.

(3) Edit "main.py". Change the parameter “lrPath” to the absolute path of “original” directory. Change the parameter “hrPath” to the absolute path of “bin” directory.

(4) Open python command line and run command "python -m visdom.server".

(5) Open the internet explorer software and enter http://localhost:8097/. The detail information of training phase will be plotted here.

(6) Run main.py and the training work start. 

### 4.Test
(1) Open test.py. 

(2) Change the parameter “root” to the absolute path of directory including test images.

(3) Change the parameter “lowMeanVal” and “lowStdVal” to the mean and standard variance values of the test images.

(4) Input the PTH file path of the codes "pretrained_net.load_state_dict(torch.load([where your pth file is],map_location='cuda:0'))". The PTH file is the python network parameter file. You can download an example PTH file from https://github.com/artzers/SuperSegmentation-Neuron/releases/tag/1.0.0. 

(5) Run test.py. 

