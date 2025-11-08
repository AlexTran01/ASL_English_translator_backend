import cv2
import time
import requests

def record_and_post(url='http://127.0.0.1:8000/translate_asl', duration=2, out='capture.mp4'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')

    # Use a common codec and size — adjust if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out, fourcc, fps, (width, height))

    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()

    with open(out, 'rb') as f:
        files = {'video': (out, f, 'video/mp4')}
        r = requests.post(url, files=files)
    print('Status:', r.status_code)
    print('Response:', r.text)

if __name__ == '__main__':
    record_and_post()