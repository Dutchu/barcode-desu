import cv2
from pyzbar import pyzbar


def detect_and_decode(frame):
    # Konwersja klatki na skale szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie konturów (prosty sposób na wykrycie paczek)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista paczek z kodami kreskowymi
    packages = []

    for contour in contours:
        # Pomijanie małych konturów, które nie są paczkami
        if cv2.contourArea(contour) < 1000:
            continue

        # Rysowanie prostokątnego obrysu wokół wykrytej paczki
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Wycięcie regionu zainteresowania (ROI) zawierającego paczkę
        roi = frame[y:y + h, x:x + w]

        # Wykrywanie kodów kreskowych w ROI
        barcodes = pyzbar.decode(roi)
        barcode_data = [barcode.data.decode("utf-8") for barcode in barcodes]

        if len(barcode_data) == 1:
            # Dodanie paczki do listy poprawnych paczek
            packages.append(barcode_data[0])
            cv2.putText(frame, barcode_data[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            # Oznaczenie paczki jako wadliwej
            cv2.putText(frame, "WADLIWA PACZKA", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, packages


def main(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, packages = detect_and_decode(frame)

        # Wyświetlenie przetworzonej klatki
        cv2.imshow('Frame', frame)

        # Przerwanie po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = 'path.mp4'  # Podaj ścieżkę do swojego wideo
    main(video_path)