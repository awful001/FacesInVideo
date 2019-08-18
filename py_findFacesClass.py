# -*- coding: utf-8 -*-

import cv2


class FacesInVideo(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def releaseCamera(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def findFacesAndEyes(self, FindEyes=False):
        ret, Image = self.cap.read()
        Image = cv2.bilateralFilter(Image, 9, 75, 75)
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        # Find faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        locations = []
        if faces is not None:
            for (x, y, w, h) in faces:
                cv2.rectangle(Image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = Image[y:y + h, x:x + w]

                # Store the location of the face
                location = [x + w // 2, y + h // 2]
                locations.append(location)

                # Find eyes
                if FindEyes:
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('Image', Image)
            # if cv2.waitKey(1) == 27:
            #     cv2.destroyAllWindows()

        return locations


# How to use:
if __name__ == '__main__':
    video1 = FacesInVideo()
    while True:
        location1 = video1.findFacesAndEyes()
        print(location1)
        if cv2.waitKey(1) == 27:
            break

    video1.releaseCamera()
