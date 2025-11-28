import cv2
import time
from .utils import calculate_fps

class VideoProcessor:
    """
    Класс для работы с потоком
    """
    
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open source: {source}")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def process_stream(self, detector, save_path=None, show_fps=True):
        out = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 20
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            detections = detector.predict(frame)
            for detection in detections:
                print(f"{detection['class']} DETECTED ({detection['conf']:.3f} confidence score) with coords: {detection['bbox']} ")
            frame = detector.draw_detections(frame, detections)

            if show_fps:
                fps = calculate_fps(prev_time)
                prev_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Drone Detection", frame)
            if out:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()
