# input: sudoku image or video
# output: auto solved sudoku

import cv2 as cv
import numpy as np
import keras
from sudoku_solver import SudokuSolver

model = keras.models.load_model('models/digit_model28x28.h5')

path = "test_images/easy.png"
size_of_digit_img = 28

raw_picture = cv.imread(path)

# xu li anh qua co
max_height, max_width = 1600, 800
height, width, _ = raw_picture.shape
print("original picture' size: " + str(width) + " x " + str(height))
if width > max_width or height > max_height:
    rate = min(max_width / width, max_height / height)
    raw_picture = cv.resize(raw_picture, dsize=None, fx=rate, fy=rate)
    height, width, _ = raw_picture.shape
    print("fix picture' size: " + str(width) + " x " + str(height))

cv.imshow("raw picture", raw_picture)
cv.waitKey()
#
raw_gray_picture = cv.cvtColor(raw_picture, cv.COLOR_BGR2GRAY)
# cv.imshow("raw gray picture", raw_gray_picture)
# cv.waitKey()

ret, raw_thresh_picture = cv.threshold(raw_gray_picture, 230, 255, cv.THRESH_BINARY)
# cv.imshow("raw threshold picture", raw_thresh_picture)
# cv.waitKey()

# tim vi tri cua sudoku's board:
contours, hierarchy = cv.findContours(raw_thresh_picture, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print("Number of all contours: " + str(len(contours)))

area = width * height
x_min, x_max, y_min, y_max = max_width + 1, -1, max_height + 1, -1
count = 0
for cnt in contours:
    if area / 9 > cv.contourArea(cnt) > 0.6 * area / 81:
        count += 1
        for point in cnt:
            x_min = min(x_min, point[0][0])
            x_max = max(x_max, point[0][0])
            y_min = min(y_min, point[0][1])
            y_max = max(y_max, point[0][1])

print("Number of squares contours: " + str(count))

print("raw picture area: " + str(area))
print(x_min, x_max, y_min, y_max)

sudoku_board = raw_picture[y_min:y_max, x_min:x_max]
# cv.imshow("Board", sudoku_board)
# cv.waitKey()

# cat 81 o vuong de nhan dien chu
board_width, board_height = x_max - x_min, y_max - y_min
print("Board's size: " + str(board_width) + " x " + str(board_height))
square_width, square_height = board_width / 9, board_height / 9
list_images = []

for i in range(9):
    for j in range(9):
        cur_img = sudoku_board[int(i*square_height)+5:int((i+1)*square_height)-5, int(j*square_width)+5:int((j+1)*square_width)-5]
        cur_img = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY)
        cur_img = cv.resize(cur_img, dsize=(size_of_digit_img, size_of_digit_img))
        cur_img = cur_img.astype("float32") / 255
        list_images.append(cur_img)

def is_blank(img, size):
    total = 0
    for arr in img:
        for dot in arr:
            total += dot
    if total > (size ** 2) * (1 - 0.02):
        return True
    return False


list_res = [0 for i in range(len(list_images))]
for i in range(len(list_images)):
    img = list_images[i]
    if is_blank(img, size_of_digit_img):
        continue

    img_np = np.array([img])
    img_np = np.expand_dims(img_np, -1)

    prediction = model.predict(img_np)
    res = np.argmax(prediction)
    list_res[i] = res

board = [[0 for _ in range(9)] for _ in range(9)]
board2 = [[0 for _ in range(9)] for _ in range(9)]
for i in range(len(list_res)):
    board[i // 9][i % 9] = list_res[i]
    board2[i // 9][i % 9] = list_res[i]

solver = SudokuSolver(board)
print("Begin sudoku: ")
solver.print()
res = solver.solver()
solver.print()

if res:
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(9):
        for j in range(9):
            if board2[i][j] != 0:
                continue
            text = str(solver.arr[i][j])
            cv.putText(sudoku_board, text, (int((j+0.5)*square_height)-15, int((i+0.5)*square_width)+15), font, 1.5, (255, 0, 0), 2, cv.LINE_AA)


cv.imshow("sudoku_solved", sudoku_board)
cv.waitKey()
cv.destroyAllWindows()