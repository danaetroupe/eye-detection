import numpy as np
import cv2, dlib

#todo: how would this program be different if the user has a lazy eye?
#if eye gets smaller, smaller flicks needed to change gaze
# if eye psoition moves, move screen
# if eye tracks then move

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks.dat")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    eyes = eye_cascade.detectMultiScale(gray)
    
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x,y), (x1,y1), (0, 0, 255)) <- face rectangle
        
        left_eye_region = np.empty((1,2), np.int32)
        landmarks = predictor(gray, face)
        for num in range(36, 42):
            landmark = landmarks.part(num)
            left_eye_region = np.append(left_eye_region, np.array([(landmark.x, landmark.y)]), axis = 0)
            #cv2.circle(frame, (landmark.x, landmark.y), 5, (0, 0, 255))
        left_eye_region = np.delete(left_eye_region, 0, axis = 0)
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)
            
        minx = np.min(left_eye_region[:, 0])
        miny = np.min(left_eye_region[:, 1])
        maxx = np.max(left_eye_region[:, 0])
        maxy = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[miny:maxy, minx:maxx]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        threshold_eye = cv2.resize(threshold_eye, None, fx=25,fy=25)
        
        cv2.imshow('Threshold', threshold_eye)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()