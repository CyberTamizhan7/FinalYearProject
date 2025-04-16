import os
import fnmatch
import cv2
import numpy as np
import time
from pdf2image import convert_from_path

# Start Time
start = time.perf_counter()

# Convert PDF to images
def pdf_to_images(pdf_file, output_folder):
    pages = convert_from_path(pdf_file, 300)
    os.makedirs(output_folder, exist_ok=True)
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        page.save(image_path, 'PNG')

# Convert matrix indices to chess square name (a1–h8)
def get_chess_square_name(row, col):
    return f"{chr(ord('a') + col)}{8 - row}"

# Resize image if too big for screen
def resize_to_screen(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# Sort corners in consistent order for perspective transform
def order_corners(corners):
    pts = np.array([c[0] for c in corners], dtype="float32")

    # Sum and diff help determine the positions
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return [top_left, top_right, bottom_right, bottom_left]

# Detect and crop chess boards
def detect_chess_boards(input_image_path, squares_folder, board_output_folder, count_start=1):
    original = cv2.imread(input_image_path)
    if original is None:
        print(f"Error: Image not found - {input_image_path}")
        return count_start

    os.makedirs(squares_folder, exist_ok=True)
    os.makedirs(board_output_folder, exist_ok=True)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding and morphology
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = original.shape[0] * original.shape[1]
    boards = []

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 0.05 * img_area:
            boards.append(approx)

    count = count_start
    for board_cnt in boards:
        corners = order_corners(board_cnt)
        if len(corners) != 4:
            continue

        # Warp board to 800x800
        dest_pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
        src_pts = np.float32(corners)
        matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
        warped = cv2.warpPerspective(original, matrix, (800, 800))

        # Save full board image
        board_image_path = os.path.join(board_output_folder, f"board_{count}.png")
        cv2.imwrite(board_image_path, warped)

        # Save squares
        board_folder = os.path.join(squares_folder, f'chess_board_{count}')
        os.makedirs(board_folder, exist_ok=True)

        square_size = 100
        for row in range(8):
            for col in range(8):
                square = warped[row*square_size:(row+1)*square_size, col*square_size:(col+1)*square_size]
                square_name = get_chess_square_name(row, col)
                square_filename = os.path.join(board_folder, f"board__{count}_{square_name}.jpg")
                cv2.imwrite(square_filename, square)

        count += 1

    return count

# === Main Execution ===
base_folder = r'C:\Users\admin\Desktop\Final Year Project'
pdf_file = os.path.join(base_folder, 'Documents', 'Greatest_551_17.pdf')
png_output_dir = os.path.join(base_folder, 'Images', 'PNGs')
chess_board_output_dir = os.path.join(base_folder, 'Images', 'Detected_Chess_Boards')
squares_output_dir = os.path.join(base_folder, 'Images', 'Squares')

# Step 1: Convert PDF pages to PNG
pdf_to_images(pdf_file, png_output_dir)

# Step 2: Process each PNG to extract chess boards and squares
image_files = fnmatch.filter(os.listdir(png_output_dir), "*.png")
image_files.sort()

board_count = 1
for filename in image_files:
    img_path = os.path.join(png_output_dir, filename)
    print(f"Processing {filename}...")
    board_count = detect_chess_boards(img_path, squares_output_dir, chess_board_output_dir, board_count)



# End Time
end = time.perf_counter()

total_seconds = end-start

hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = total_seconds % 60

print(f"Total Running Time: {hours}hours {minutes}minutes {seconds:.2f}seconds")
    
