from ultralytics import YOLO
import cv2
import json
from utils import format_detection


def process_video(video_path="videos/video.mp4", output_json="output.json"):
    model = YOLO("yolov5su.pt")  # Load YOLOv5 pre-trained model
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    else:
        print(f"Successfully opened video file {video_path}")

    results_list = []

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to read frame or end of video reached.")
            break

        # Run inference on the frame
        results = model.predict(frame, device='cpu')
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result[:6]
            detection = format_detection(
                object_name="person",
                object_id=1,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                subobject_name="helmet",
                subobject_id=1,
                subobject_bbox=[int(x1) + 10, int(y1) + 10, int(x2) - 10, int(y2) - 10]
            )
            results_list.append(detection)

        # Display the frame
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.resizeWindow("Video", 800, 600)         # Set window size (optional)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Save results as JSON
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=4)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video("videos/video.mp4")
