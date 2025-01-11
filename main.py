from ultralytics import YOLO
import cv2
import json
from utils import format_detection, save_cropped_image

def process_video(video_path, output_json="output.json"):
    model = YOLO("yolov5s.pt")  # Load YOLOv5 pre-trained model
    video = cv2.VideoCapture(video_path)
    results_list = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame, device='cpu')  # Run inference on CPU
        for result in results.xyxy[0]:  # Iterate through detections
            x1, y1, x2, y2, conf, class_id = result[:6]
            # Example: Add a dummy sub-object
            detection = format_detection(
                object_name="person",
                object_id=1,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                subobject_name="helmet",
                subobject_id=1,
                subobject_bbox=[int(x1) + 10, int(y1) + 10, int(x2) - 10, int(y2) - 10]
            )
            results_list.append(detection)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Save results as JSON
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=4)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("sample_video.mp4")

