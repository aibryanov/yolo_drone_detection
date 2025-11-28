from ultralytics import YOLO
import cv2
import torch


class YOLODetector:
    """
    Класс модели YOLO
    """
    
    def __init__(self, model_path: str, device: str, conf_thres: float = 0.2):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model on device: {device}")

        self.device = device
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def predict(self, frame):
        """Прогон кадра через модель YOLO"""
        results = self.model.predict(frame, verbose=False, device=self.device)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = box.conf[0].item()
                if conf < self.conf_thres:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": "drone",
                    "conf": conf
                })
                print()
        return detections

    def draw_detections(self, frame, detections):
        """Отрисовка bbox и confidence"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
