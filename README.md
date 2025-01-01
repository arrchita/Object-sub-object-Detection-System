# YOLO Object Detection in Video

This project performs object detection using a YOLOv5 model on a video. It reads frames from a video file, runs YOLOv5 inference on each frame, and stores the detected objects in a JSON format.

## Features

- Loads and processes a video frame by frame.
- Applies YOLOv5 for object detection on each frame.
- Extracts and stores object detection results in a structured format (JSON).
- Option to skip frames for faster processing.
- Displays the video in a resizable window during processing.

## Requirements

- Python 3.x
- OpenCV
- YOLOv5

### Installing Dependencies

To get started, clone the repository and install the required dependencies.

```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```

### Dependencies List

You can install the required dependencies manually using:

```bash
pip install opencv-python
pip install yolov5
```

## Usage

### 1. Prepare Your Video

Make sure you have a video file ready to use. The default video file in the code is named `sample_video.mp4`. Replace this with your own video path if needed.

### 2. Run the Script

Run the `main.py` script to start the object detection process. This will process the video frame by frame and output the results in a JSON file.

```bash
python main.py
```

- The script will process the video located at `sample_video.mp4` by default.
- The detected objects will be saved in `output.json`.
- The video will be displayed in a resizable window during processing.

## Known Issues

- **Error with Frame Reading:**
  The program may still throw an error when trying to read frames or reach the end of the video. The message "Failed to read frame or end of video reached" will appear, indicating that the video could not be processed correctly. This could be due to video file compatibility, large video size, or issues with the `cv2.VideoCapture` function.

  If you encounter this error, try using smaller video files or experimenting with different video formats.
  
PS: tons of errors- mine still does not work :(
