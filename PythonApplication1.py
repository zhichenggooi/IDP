import cv2
from ultralytics import YOLO

def detect_objects():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("YOLOv8 Detection started. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break

        results = model(frame, stream=True)

        for result in results:
            annotated_frame = result.plot()

        cv2.imshow("YOLOv8 Bounding Boxes", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()