import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import dilation, disk
import pytesseract
import time

sudoku_grid = []
sudoku_flatten_array = []
binary_array = []
original_img = cv2.imread("images/sudoku.png", cv2.IMREAD_GRAYSCALE)
img_cropped = original_img
img_cropped_copy = original_img


def process_img():
    # 1. GaussianBlur to reduce noise obtained in thresholding algorithm
    processed_img = cv2.GaussianBlur(original_img.copy(), (9, 9), 0)
    # 2. Threshold (segmentation) and invert colors
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    # 3. Dilation to increase thickness
    processed_img = dilation(processed_img, disk(1))
    return processed_img


def divide_to_cells(processed_img):
    global sudoku_grid
    global img_cropped
    global img_cropped_copy

    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 90000:  # outer border: 90775.5
            print("Outer contour: ", area)
            cv2.drawContours(original_img, cnt, -1, (0, 0, 0), 5)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            width, height = 900, 900
            pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_cropped = cv2.warpPerspective(original_img, matrix, (width, height))
            img_cropped_copy = img_cropped.copy()
            # binary image
            _, img_cropped = cv2.threshold(img_cropped, 125, 255, cv2.THRESH_BINARY)

            # crop cells
            number_array = []
            for y in range(1, 10):
                for x in range(1, 10):
                    cell = img_cropped[y * 100 - 100:y * 100, x * 100 - 100:x * 100]
                    # removing borders
                    for cell_x in range(0, 100):
                        for cell_y in range(0, 100):
                            if (cell_x < 15 or cell_x > 85) or (cell_y < 15 or cell_y > 85):
                                cell[cell_x][cell_y] = 255
                    # get a string value from picture
                    value = pytesseract.image_to_string(cell, config='--psm 6')  # --psm 6 - assume a single uniform block of text.
                    if not value:
                        value = 0
                        binary_array.append(1)
                    else:
                        value = int(value[0])  # output before: i.e. '7\n'
                        binary_array.append(0)
                    number_array.append(value)
            sudoku_grid = np.reshape(number_array, (9, 9))
            print("Read grid:\n", sudoku_grid, "\n")


def possible(y, x, n):
    for i in range(0, 9):
        if sudoku_grid[y][i] == n:
            return False
    for i in range(0, 9):
        if sudoku_grid[i][x] == n:
            return False
    x0 = (x // 3) * 3
    y0 = (y // 3) * 3

    for i in range(0, 3):
        for j in range(0, 3):
            if sudoku_grid[y0 + i][x0 + j] == n:
                return False
    return True


def solve():
    global sudoku_grid

    for y in range(0, 9):
        for x in range(0, 9):
            if sudoku_grid[y][x] == 0:
                for n in range(1, 10):
                    if possible(y, x, n):
                        sudoku_grid[y][x] = n
                        solve()
                        sudoku_grid[y][x] = 0
                return
    sudoku_flatten_array = np.array(sudoku_grid).flatten()
    solved_numbers = sudoku_flatten_array * binary_array  # to insert number only in empty cells
    display(img_cropped_copy, solved_numbers)
    print("Solved Sudoku grid:")
    print(sudoku_grid)


def display(img, numbers):
    width = 100
    height = 100
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y * 9) + x]),
                            (x * width + int(width / 2) - 25, int((y + 0.75) * height)), cv2.FONT_HERSHEY_DUPLEX,
                            2, (0, 0, 0), 5)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title('Sudoku Solver')
    ax.axis('off')
    fig.set_tight_layout(tight=True)
    plt.show()


def main():
    start = time.time()
    print("\nProgram is running...\n")
    img = process_img()
    divide_to_cells(img)
    solve()
    end = time.time()
    print("\nProgram finished. \nTime: {:.2f}".format(end - start), "s.")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()
