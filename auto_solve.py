import cv2 as cv
import numpy as np
import keras
from sudoku_solver import SudokuSolver
import sys

def load_picture(filename):
    return cv.imread(filename)

def load_model(filename):
    model = keras.models.load_model(filename)
    return model

def preprocessing_images(raw_pic, show=False):
    max_height, max_width = 1600, 800
    height, width, _ = raw_pic.shape
    fixed_pic = raw_pic
    print("original picture' size: " + str(width) + " x " + str(height))
    if width > max_width or height > max_height:
        rate = min(max_width / width, max_height / height)
        fixed_pic = cv.resize(raw_pic, dsize=None, fx=rate, fy=rate)
        height, width, _ = fixed_pic.shape
        print("fix picture' size: " + str(width) + " x " + str(height))
    if show:
        cv.imshow("raw picture", raw_pic)
        cv.waitKey()
    return fixed_pic

def to_gray_scale(colored_pic, show=False):
    gray_pic = cv.cvtColor(colored_pic, cv.COLOR_BGR2GRAY)
    if show:
        cv.imshow("gray picture", gray_pic)
        cv.waitKey()
    return gray_pic

def to_binary_scale(gray_pic, show=False):
    ret, binary_pic = cv.threshold(gray_pic, 230, 255, cv.THRESH_BINARY)
    if show:
        cv.imshow("binary picture", binary_pic)
        cv.waitKey()
    return binary_pic

def find_box(binary_pic):
    contours, hierarchy = cv.findContours(binary_pic, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("Number of all contours: " + str(len(contours)))
    height, width = binary_pic.shape
    area = width * height

    x_min, x_max, y_min, y_max = width + 1, -1, height + 1, -1
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
    return x_min, x_max, y_min, y_max

def get_sudoku_board(fixed_pic, box, show=False):
    x_min, x_max, y_min, y_max = box
    sudoku_board = fixed_pic[y_min:y_max, x_min:x_max]
    if show:
        cv.imshow("Board", sudoku_board)
        cv.waitKey()
    return sudoku_board

def split_squares(sudoku_board):
    board_height, board_width, _ = sudoku_board.shape
    print("Board's size: " + str(board_width) + " x " + str(board_height))
    square_width, square_height = board_width / 9, board_height / 9
    list_images = []

    for i in range(9):
        for j in range(9):
            img = sudoku_board[int(i*square_height)+5:int((i+1)*square_height)-5, int(j*square_width)+5:int((j+1)*square_width)-5]
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, dsize=(size_of_digit_img, size_of_digit_img))
            img = img.astype("float32") / 255
            list_images.append(img)

    return list_images

def is_blank(img, size):
    total = 0
    for arr in img:
        for dot in arr:
            total += dot
    if total > (size ** 2) * (1 - 0.02):
        return True
    return False

def get_num_list(list_images, model):
    num_list = [0 for _ in range(len(list_images))]
    for i in range(len(list_images)):
        img = list_images[i]
        if is_blank(img, size_of_digit_img):
            continue
        img_np = np.array([img])
        img_np = np.expand_dims(img_np, -1)
        prediction = model.predict(img_np)
        res = np.argmax(prediction)
        num_list[i] = res
    return num_list

def normalize_board(num_list):
    num_board = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(len(num_list)):
        num_board[i // 9][i % 9] = num_list[i]
    return num_board

def solve_sudoku(blank_sudoku_board, num_board, num_board2, show=False):
    board_height, board_width, _ = blank_sudoku_board.shape
    square_width, square_height = board_width / 9, board_height / 9

    solver = SudokuSolver(num_board)
    print("Begin sudoku: ")
    solver.print()
    res, answer_arr = solver.solver()
    solver.print()
    if res:
        font = cv.FONT_HERSHEY_SIMPLEX
        for i in range(9):
            for j in range(9):
                if num_board2[i][j] != 0:
                    continue
                text = str(answer_arr[i][j])
                blank_sudoku_board = cv.putText(blank_sudoku_board, text,
                                                (int((j+0.5)*square_height)-15, int((i+0.5)*square_width)+15),
                                                font, 1.5, (255, 0, 0), 2, cv.LINE_AA)
    if show:
        cv.imshow("sudoku_solved", blank_sudoku_board)
        cv.waitKey()
    return blank_sudoku_board, answer_arr


if __name__ == '__main__':
    path = 'test_images/easy.png'
    model_path = 'models/digit_model28x28.h5'
    size_of_digit_img = 28

    args = sys.argv[1:]
    if args:
        path = args[0].replace("'", '')

    raw_picture = load_picture(path)
    if raw_picture is not None:
        model = load_model(model_path)
        fixed_picture = preprocessing_images(raw_picture, show=True)
        gray_picture = to_gray_scale(fixed_picture, show=False)
        binary_picture = to_binary_scale(gray_picture, show=False)
        bounding_box = find_box(binary_picture)
        sudoku_board = get_sudoku_board(fixed_picture, bounding_box, show=False)
        squares_img_list = split_squares(sudoku_board)
        squares_num_list = get_num_list(squares_img_list, model)
        norm_board, norm_board2 = normalize_board(squares_num_list), normalize_board(squares_num_list)

        solve_sudoku(sudoku_board, norm_board, norm_board2, show=True)
    else:
        print('Path not exist!')

    cv.destroyAllWindows()