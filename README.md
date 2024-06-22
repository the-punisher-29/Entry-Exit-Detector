# Entry-Exit-Detector

- Use case: counting the number of people in the stores/buildings/shopping malls etc., in real-time.
- Automating features and optimising the real-time stream for better performance (with threading).
- Acts as a measure towards footfall analysis and in a way to tackle COVID-19 scenarios.

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
To run on a webcam, set ```"url": 0``` in ```utils/config.json``` and run the command:

```
python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel
```



Suggestions are welcome!

If you found any issue,Please raise the same.
