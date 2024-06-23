import cv2
import numpy as np
from pyzbar import pyzbar


def preprocess_frame(frame):
    # Konwersja do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wyrównanie histogramu w celu poprawienia kontrastu
    gray = cv2.equalizeHist(gray)

    # Adaptacyjne progowanie, aby zminimalizować wpływ zmiany oświetlenia
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Operacje morfologiczne: otwarcie, aby usunąć szumy
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return morph


def detect_packages(frame, processed_frame):
    # Znalezienie konturów w przetworzonym obrazie
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    package_count = 0
    for cnt in contours:
        # Pomijanie zbyt małych konturów, które mogą być szumami
        if cv2.contourArea(cnt) > 17000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            package_count += 1

    return package_count


def decode_barcodes(frame):
    barcodes = pyzbar.decode(frame)
    decoded_barcodes = []

    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        if barcode_type == "PDF417":
            continue

        decoded_barcodes.append(barcode_data)

        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, decoded_barcodes


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    total_packages = 0
    scanned_barcodes = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        package_count = detect_packages(frame, processed_frame)
        total_packages += package_count

        frame, decoded_barcodes = decode_barcodes(frame)

        for barcode in decoded_barcodes:
            if barcode not in scanned_barcodes:
                scanned_barcodes.add(barcode)
                print(f"Zeskanowano nowy kod kreskowy: {barcode}")

        cv2.imshow('Original', frame)
        cv2.imshow('Processed', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total packages detected: {total_packages}")
    print(f"Łączna liczba unikalnych zeskanowanych kodów kreskowych: {len(scanned_barcodes)}")


if __name__ == "__main__":
    video_path = 'film.MOV'  # Zamień na ścieżkę do swojego pliku wideo
    main(video_path)
