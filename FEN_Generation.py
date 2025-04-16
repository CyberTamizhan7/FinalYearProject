import os
import cv2
import json
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime


# Start Time
start = time.perf_counter()


today_date = datetime.today().strftime('%Y.%m.%d')


# === Load the model ===
model = load_model("Chess_22_20.h5")

# === Load class labels ===
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# === Image preprocessing function ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Prediction function ===
def predict_piece(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_label = class_labels[class_index]
    return predicted_label, confidence * 100

# === FEN compression ===
def compress_fen_row(row):
    result = ''
    count = 0
    for ch in row:
        if ch == '#':
            count += 1
        else:
            if count > 0:
                result += str(count)
                count = 0
            result += ch
    if count > 0:
        result += str(count)
    return result

# === Base path for all folders ===
base_path = r"C:\Users\admin\Desktop\Final Year Project\Images\Squares"

# === Pieces dictionary ===
piecesDict = {
    "black_king":"k", "black_queen":"q", "black_rook":"r", "black_bishop":"b", "black_knight":"n", "black_pawn":"p",
    "white_king":"K", "white_queen":"Q", "white_rook":"R", "white_bishop":"B", "white_knight":"N", "white_pawn":"P"
}

# === Output PGN file ===
final_project_path = os.path.dirname(os.path.dirname(base_path))
output_path = os.path.join(final_project_path, "positions.pgn")

with open(output_path, "w") as pgn_file:

    # Loop through chess_board_1 to chess_board_722
    for board_num in range(1, 10):
        folder_path = os.path.join(base_path, f"chess_board_{board_num}")
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        print(f"\n📂 Processing: {folder_path}")
        fen = ""

        # Process images in sorted order
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                predicted_piece, confidence = predict_piece(image_path)
                print(f"{filename}: {predicted_piece} ({confidence:.2f}%)")

                if predicted_piece.lower() in ["black_blank", "white_blank"]:
                    fen += "#"
                elif predicted_piece.lower() in piecesDict:
                    fen += piecesDict[predicted_piece.lower()]
                else:
                    print(f"⚠️ Unknown label: {predicted_piece}")

        # Convert to 8x8 and transpose
        fen_temp_list = [fen[i:i+8] for i in range(0, 64, 8)]
        reversed_FEN = [row[::-1] for row in fen_temp_list]
        transposed_FEN = [''.join(row[i] for row in reversed_FEN) for i in range(8)]
        compressed_FEN = [compress_fen_row(row) for row in transposed_FEN]
        fen_string = '/'.join(compressed_FEN)

        # Write to PGN
        pgn_file.write(f'[Event "Chess Position"]\n')
        pgn_file.write(f'[Date "{today_date}"]\n')
        pgn_file.write(f'[White "Position {board_num}"]\n')
        pgn_file.write(f'[FEN "{fen_string} b"]\n\n')
        pgn_file.write("*\n\n")

print(f"\n✅ All positions saved to: {output_path}")





# End Time
end = time.perf_counter()

total_seconds = end-start

hours = int(total_seconds//3600)
minutes = int((total_seconds%3600)//60)
seconds = int(total_seconds%60)

print(f"Total Running Time : {hours}hours {minutes}minutes {seconds:.2f}seconds")
