# SuperSegmentation-Neuron

### Requirement
Our Training and Test environment is Python3.7.3, Pytorch 1.5.1+Cuda 9.2. 
Other python packages:
- Numpy
- tqdm
- tifffile
- visdom
- scanf

### Prepare
1. Use Other neurite tracing tools such like NeuroGPS-Tree(https://github.com/GTreeSoftware/GTree), Vaa3d(https://github.com/Vaa3D) to trace the neurites. The traced neurites are saved in SWC files. The swc files from the same neuron are saved into one directory.
2. Edit MergeSwc.py. Change the srcPath and the path of dirList to your real data path. Run  MergeSwc.py to generate the %_allSwc.swc.
3. Edit the SampleMaker.py. Change the srcPath and the path of dirList to your real data path. Run SampleMaker.py to generate the binary image as the gold truth.

### Training
1. Put the original images into "original" directory and binary images into "bin" directory.
2. Open "main.py" and turn to line 79. Change the 1st and 2nd parameters of DataPackage. The 1st parameter is the directory of original directory. The 2nd parameter is the directory of bin directory.
3. Run main.py

### Test
Open test.py. Turn to line 8, change the path of test images. Then turn to line 12 and change the mean and std values. Run test.py.
