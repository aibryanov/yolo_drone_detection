import time


def calculate_fps(prev_time):
    """
    Вычисляет FPS по времени предыдущего кадра.
    """
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time + 1e-6)
    return fps