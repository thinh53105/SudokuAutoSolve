# input: arr[9][9]
# output: solved arr[9][9]


class SudokuSolver(object):

    def __init__(self, arr):
        self.arr = arr

    def get_ava_number(self, row, col):
        if self.arr[row][col] != 0:
            return None
        tmp = [i for i in range(1, 10)]
        for c in range(9):
            if self.arr[row][c] in tmp:
                tmp.remove(self.arr[row][c])
        for r in range(9):
            if self.arr[r][col] in tmp:
                tmp.remove(self.arr[r][col])
        row_box = row // 3
        col_box = col // 3
        for r in range(3*row_box, 3*row_box + 3):
            for c in range(3*col_box, 3*col_box + 3):
                if self.arr[r][c] in tmp:
                    tmp.remove(self.arr[r][c])
        return tmp

    def find_blank(self, row, col):
        num = 9*row + col
        for i in range(num, 81):
            next_row = i // 9
            next_col = i % 9
            if self.arr[next_row][next_col] == 0:
                return next_row, next_col
        return None

    def solve(self, cur_row, cur_col):
        if not self.find_blank(cur_row, cur_col):
            return True
        next_row, next_col = self.find_blank(cur_row, cur_col)
        ava_list = self.get_ava_number(next_row, next_col)
        for number in ava_list:
            self.arr[next_row][next_col] = number
            if self.solve(next_row, next_col):
                return True
            self.arr[next_row][next_col] = 0
        return False

    def solver(self):
        print("solving...")
        res = self.solve(0, 0)
        print("done.")
        return res

    def print(self):
        for row in self.arr:
            print(row)