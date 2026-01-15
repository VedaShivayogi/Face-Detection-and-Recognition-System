import cv2
import os
from time import time
from tkinter import messagebox, simpledialog

def main_app(name=None, timeout=5):
    # -------- DEFAULT NAME OR ASK USER --------
    if name is None or str(name).strip() == "":
        name = simpledialog.askstring("Input", "Enter person name:")
        if not name:
            messagebox.showerror("Error", "No name provided!")
            return

    print("Detector using name:", name)

    face_cascade_path = "./data/haarcascade_frontalface_default.xml"
    classifier_path = f"./data/classifiers/{name}_classifier.xml"

    if not os.path.exists(classifier_path):
        choice = messagebox.askyesno(
            "Classifier Missing",
            f"Classifier for '{name}' not found.\nDo you want to train new classifier?"
        )
        if choice:
            messagebox.showinfo("Info", f"Please train classifier for '{name}' first.")
        else:
            messagebox.showinfo("Info", "Operation cancelled.")
        return

    # Load classifier
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(classifier_path)

    cap = cv2.VideoCapture(0)
    pred = False
    start_time = time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                _, confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
            except:
                confidence = 0

            if confidence > 50:
                pred = True
                text = f"Recognized: {name.upper()}"
                color = (0, 255, 0)
            else:
                text = "Unknown Face"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-5),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        if time() - start_time >= timeout:
            break

    cap.release()
    cv2.destroyAllWindows()

    if pred:
        messagebox.showinfo("Congrat", "You have already checked in")
    else:
        messagebox.showerror("Alert", "Please check in again")
