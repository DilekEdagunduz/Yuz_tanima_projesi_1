

import cv2
import numpy as np


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('egitim/egitim.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# id sayacını başlat
id = 0
names = ['None', 'Dilek Eda','Gunduz']
# Canlı video yakalamayı başlat
kamera = cv2.VideoCapture(0)
kamera.set(3, 1000)  # video genişliğini belirle
kamera.set(4, 800)  # video yüksekliğini belirle
# minimum pencere boyutunu belirle
minW = 0.1 * kamera.get(3)  # genişlik
minH = 0.1 * kamera.get(4)  # yükseklik
while True:
    ret, img = kamera.read()
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yuzler = faceCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, uyum = recognizer.predict(gri[y:y + h, x:x + w])

        if (uyum < 100):
            id = names
            uyum = f"Uyum=  {round(uyum,0)}%"
        else:
            id = "bilinmiyor"
            uyum = f"Uyum=  {round(uyum,0)}%"

        color = (255,255,255)

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(uyum), (x + 5, y + h + 25), font, 1, (255, 255, 0), 1)

    cv2.imshow('kamera', img)
    k = cv2.waitKey(10) & 0xff  # Çıkış için Esc veya q tuşu
    if k == 27 or k==ord('q'):
        break
# Belleği temizle
print("\n [INFO] Programdan çıkıyor ve ortalığı temizliyorum")
kamera.release()
cv2.destroyAllWindows()