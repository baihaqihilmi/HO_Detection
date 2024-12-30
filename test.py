import cv2
import argparse
import os.path as osp
import time
from models.inference import OpenVinoInference

def parse_args():
    parser = argparse.ArgumentParser(description="Video Capture Test")
    parser.add_argument('--source', type=int, default=2, help='Path to the video source')
    parser.add_argument('--version', type=str, default='v1', help='Path to the video source')
    return parser.parse_args()

def main():
    args = parse_args()
    model = OpenVinoInference(model_path=osp.join('models', args.version))
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Perform inference
        result, bbox = model(frame)

        # Display FPS on the frame
        cv2.putText(result, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()