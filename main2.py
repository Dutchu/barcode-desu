import cv2
from pyzbar import pyzbar


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


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    scanned_barcodes = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, decoded_barcodes = decode_barcodes(frame)

        for barcode in decoded_barcodes:
            if barcode not in scanned_barcodes:
                scanned_barcodes.add(barcode)
                print(f"Zeskanowano nowy kod kreskowy: {barcode}")

        cv2.imshow('Barcode Scanner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Łączna liczba unikalnych zeskanowanych kodów kreskowych: {len(scanned_barcodes)}")


if __name__ == "__main__":
    video_path = 'film.MOV'  # Zamień na ścieżkę do swojego pliku wideo
    process_video(video_path)
