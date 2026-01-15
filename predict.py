import cv2
import os

def predict(name, video_path=0):  # 0 = webcam, else video file path
    # Face detection cascade
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    
    # Face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    classifier_file = f"./data/classifiers/{name}_classifier.xml"
    if not os.path.exists(classifier_file):
        print(f"Classifier file not found for {name}")
        return
    recognizer.read(classifier_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video/webcam!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                _, conf = recognizer.predict(roi)
                conf = 100 - int(conf)
            except:
                conf = 0
            
            if conf > 50:
                text = f"{name.upper()} Recognized"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage examples:
# Webcam
# predict("tho", 0)

# Video file
predict("tho", r"data\WIN_20230920_07_56_11_Pro.mp4")
