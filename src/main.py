import cv2
from detector import MotionDetector

def main():
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera bulunamadı!")
        return

    # Modeli başlat
    detector = MotionDetector()

    print("🎥 Gerçek zamanlı ani hareket tespiti başladı.")
    print("Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı.")
            break

        # Görüntüyü yatay çevir (doğal görünüm)
        frame = cv2.flip(frame, 1)

        # Model tahmini
        status = detector.predict(frame)

        # Renk seçimi
        if status == "Normal":
            color = (0, 255, 0)
        elif status == "Ani Hareket":
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        # Ekrana yazı yaz
        cv2.putText(frame, f"Durum: {status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Görüntüyü göster
        cv2.imshow("Ani Hareket Tespiti", frame)

        # 'q' ile çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
