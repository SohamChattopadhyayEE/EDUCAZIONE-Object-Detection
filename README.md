# Object Detection
This is a real-time / video object detection project using `yolov3` model. The entire project is in Python programming language.
## Dependencies
- install packages : `.\directory\object_detection> pip install requirements.txt`
- download yolov3 weights :  `.\directory\object_detection\model_yolov3> download_weights.txt `
## Arguments
    .\directory\object_detection> python object_detection.py --help
    usage: object_detection.py [-h] [-wt MODEL_WEIGHTS] [-cfg MODEL_CFG] [-cls DATASET_CLASSES] [-is_realtime] [-vid VID_PATH]
                               [-fd FRAME_DIMENSION] [-conf_threh THRESHOLD_CONFIDENCE]

    Object detection with OpenCV

    optional arguments:
      -h, --help            show this help message and exit
      -wt MODEL_WEIGHTS, --model_weights MODEL_WEIGHTS
                            Path to the pre-trained weights of the network
      -cfg MODEL_CFG, --model_cfg MODEL_CFG
                            Path to the .config file of the network
      -cls DATASET_CLASSES, --dataset_classes DATASET_CLASSES
                            Path to the classes file of the dataset
      -is_realtime, --is_realtime
                            Is real time object detection is being performed
      -vid VID_PATH, --vid_path VID_PATH
                            Path to the video file
      -fd FRAME_DIMENSION, --frame_dimension FRAME_DIMENSION
                            The dimensions of the frames
      -conf_threh THRESHOLD_CONFIDENCE, --threshold_confidence THRESHOLD_CONFIDENCE
                            Threshold confidence for selecting an object
