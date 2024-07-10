# Entry-Exit-Detector

- Use case: counting the number of people in the stores/buildings/shopping malls etc., in real-time.
- Automating features and optimising the real-time stream for better performance (with threading).
- Acts as a measure towards footfall analysis

## Simple Theory

### SSD detector
- We are using a SSD ```Single Shot Detector``` with a MobileNet architecture. In general, it only takes a single shot to detect whatever is in an image. That is, one for generating region proposals, one for detecting the object of each proposal. 
- Compared to other two shot detectors like R-CNN, SSD is quite fast.
- ```MobileNet```, as the name implies, is a DNN designed to run on resource constrained devices. For e.g., mobiles, ip cameras, scanners etc.
- Thus, SSD seasoned with a MobileNet should theoretically result in a faster, more efficient object detector.

### Centroid tracker
- Centroid tracker is one of the most reliable trackers out there.
- To be straightforward, the centroid tracker computes the ```centroid``` of the bounding boxes.
- That is, the bounding boxes are ```(x, y)``` co-ordinates of the objects in an image. 
- Once the co-ordinates are obtained by our SSD, the tracker computes the centroid (center) of the box. In other words, the center of an object.
- Then an ```unique ID``` is assigned to every particular object deteced, for tracking over the sequence of frames.

## Why MobileNet + SSD
In this project, we harness the power of MobileNet and Single Shot Detector (SSD) architectures to achieve blazing-fast, real-time object detection on resource-limited devices like Raspberry Pi, smartphones, and more.

Single Shot Detectors (SSDs) excel by predicting bounding boxes and class probabilities directly from feature maps in a single pass, making them perfect for real-time applications. We pair SSDs with MobileNets, lightweight convolutional neural networks designed for mobile and embedded vision applications. MobileNets are super efficient due to their use of depthwise separable convolutions, drastically reducing the number of parameters and computations compared to traditional convolutional networks.

Combining MobileNet with SSD strikes a balance between speed and accuracy, making this approach ideal for real-time object detection on devices with limited computational power.

## Technical Implementation:

OpenCV's dnn module is used to load a pre-trained Caffe implementation of MobileNet SSD, originally trained on COCO and fine-tuned on PASCAL VOC, achieving a mean average precision (mAP) of 72.7%.Input images are fed into the network to obtain bounding box coordinates for each detected object.

### Advantages of SSDs:
- SSDs offer a sweet spot between speed and accuracy, making them preferable to Faster R-CNNs (complex and slower) or YOLO (fast but less accurate).
- SSDs provide a simpler and well-documented approach with a faster FPS throughput compared to YOLO.

### Why MobileNets:

- Designed for resource-constrained devices, MobileNets are far smaller (200-500MB) than traditional architectures like VGG or ResNet.
- Depthwise separable convolutions reduce parameters and computations, making MobileNets more resource-efficient.
- By joining forces with MobileNets, SSDs become even more efficient and effective, enabling real-time object detection on your favorite devices.

```Credit: The MobileNet SSD model used in this project was trained by chuanqi305 (https://github.com/chuanqi305/MobileNet-SSD).```

## Steps to Run the Model-

### Install the dependencies
First up, install all the required Python dependencies by running: ```
pip install -r requirements.txt ```

### In VS Code Terminal

#### Test video file
To run inference on a test video file, head into the root directory and run the command: 
```
python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel --input utils/data/tests/test_1/2.mp4
```

#### Webcam
To run on a webcam,run the command:

```
python pc_webcam.py
```



Suggestions are welcome!

If you found any issue,Please raise the same.
