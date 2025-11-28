import argparse
from src.detection import YOLODetector
from src.video_processor import VideoProcessor
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="webcam",
                        help="webcam / path to video / path to image")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save video")
    parser.add_argument("--showfps", action="store_true", help="Show FPS")
    parser.add_argument("--no-showfps", action="store_false", dest="showfps", help="Do not show FPS")
    parser.add_argument("--device", type=str, default=None, help="Device for inference")
    args = parser.parse_args()

    detector = YOLODetector(args.weights, device=args.device, conf_thres=0.25)

    if args.source.lower() == "webcam":
        source = 0
    else:
        source = args.source

    if str(source).lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        frame = cv2.imread(source)
        detections = detector.predict(frame)
        frame = detector.draw_detections(frame, detections)
        cv2.imshow("Drone Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        processor = VideoProcessor(source)
        processor.process_stream(detector, save_path=args.save, show_fps=args.showfps)

if __name__ == "__main__":
    main()
