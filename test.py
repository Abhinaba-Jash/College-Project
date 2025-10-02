import cv2
import numpy as np
from keras.models import load_model

def predict_video(video_path, model, frame_sample=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and len(frames) < frame_sample:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            frame = cv2.resize(frame, (256, 256))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
        count += 1
    cap.release()

    frames = np.array(frames)
    preds = model.predict(frames)
    mean_pred = np.mean(preds)
    return "FAKE" if mean_pred > 0.5 else "REAL"

if __name__ == "__main__":
    model = load_model("mesonet_model.h5")
    test_video = "data/test_videos/sample.mp4"
    result = predict_video(test_video, model)
    print(f"🎬 Video: {test_video} → Prediction: {result}")
