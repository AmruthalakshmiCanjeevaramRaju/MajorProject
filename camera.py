import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd

# Load the emotion detection model
emotion_model = load_model('modelvgg.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define emotion labels
emotion_dict = {0: "angry", 1: "happy", 2: "sad", 3: "surprise"}

# Define music distribution paths
music_dist = {
    0: "songs/angry.csv",
    1: "songs/happy.csv",
    2: "songs/sad.csv",
    3: "songs/surprise.csv"
}

# Function for reading video stream, generating prediction, and recommendations
class VideoCamera:
    def _init_(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.show_text = [0]

    def get_frame(self):
        ret, image = self.cap.read()
        if not ret:
            return None, None

        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        df1 = self.music_rec()  # Call the music_rec method
        df1 = pd.read_csv(music_dist[show_text[0]])
        df1 = df1[['Name','Album','Artist']]
        df1 = df1.head(15)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            self.show_text[0] = maxindex
            cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255),
                        2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        return frame, df1

    def music_rec(self):
     try:
        df = pd.read_csv(music_dist[self.show_text[0]])
        df = df[['Name', 'Album', 'Artist']]
        df = df.head(15)
        return df
     except FileNotFoundError:
        print("CSV file not found for the emotion:", emotion_dict[self.show_text[0]])
        return None

    def release(self):
        self.cap.release()

# Main function to run the application
def main():
    camera = VideoCamera()

    while True:
        frame, df1 = camera.get_frame()
        if frame is None or df1 is None:
            break

        # Display frame and recommendations if available
        cv2.imshow('Emotion Recognition', cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), -1))
        
        # Check if df1 is None before calling head()
        if df1 is not None:
            print(df1.head(15))  # Print recommendations if available
        else:
            print("No music recommendations available.")
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if _name_ == '_main_':
    main()
