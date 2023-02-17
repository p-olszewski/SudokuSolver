import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import time

amount_of_images = 243
images_array = []
grid1 = [['0', '0', '0', '0', '7', '0', '0', '0', '0'],
         ['0', '0', '6', '0', '0', '0', '7', '0', '0'],
         ['2', '0', '0', '8', '0', '3', '0', '0', '5'],
         ['0', '0', '8', '0', '0', '0', '5', '0', '0'],
         ['0', '2', '0', '4', '0', '9', '0', '3', '0'],
         ['9', '0', '0', '6', '0', '7', '0', '0', '2'],
         ['5', '0', '9', '0', '0', '0', '3', '0', '8'],
         ['0', '0', '3', '0', '0', '0', '9', '0', '0'],
         ['0', '7', '0', '9', '0', '4', '0', '5', '0']]

grid2 = [['8', '0', '0', '0', '1', '0', '0', '0', '9'],
         ['0', '5', '0', '8', '0', '7', '0', '1', '0'],
         ['0', '0', '4', '0', '9', '0', '7', '0', '0'],
         ['0', '6', '0', '7', '0', '1', '0', '2', '0'],
         ['5', '0', '8', '0', '6', '0', '1', '0', '7'],
         ['0', '1', '0', '5', '0', '2', '0', '9', '0'],
         ['0', '0', '7', '0', '4', '0', '6', '0', '0'],
         ['0', '8', '0', '3', '0', '9', '0', '4', '0'],
         ['3', '0', '0', '0', '5', '0', '0', '0', '8']]

grid3 = [['0', '0', '7', '4', '0', '9', '5', '0', '0'],
         ['0', '2', '0', '0', '7', '0', '0', '1', '0'],
         ['4', '0', '0', '0', '0', '0', '0', '0', '3'],
         ['1', '0', '0', '0', '8', '0', '0', '0', '2'],
         ['6', '0', '0', '5', '0', '3', '0', '0', '9'],
         ['0', '5', '0', '0', '2', '0', '0', '4', '0'],
         ['0', '0', '4', '0', '0', '0', '6', '0', '0'],
         ['0', '0', '0', '2', '0', '8', '0', '0', '0'],
         ['0', '0', '0', '0', '5', '0', '0', '0', '0']]


def get_images_array():
    img_array = [cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE), cv2.imread("test1.jpeg", cv2.IMREAD_GRAYSCALE), cv2.imread("test2.jpg", cv2.IMREAD_GRAYSCALE)]
    contours = []

    # preprocess images
    for i in range(0, 3):
        img_array[i] = cv2.GaussianBlur(img_array[i].copy(), (9, 9), 0)
        img_array[i] = cv2.adaptiveThreshold(img_array[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        contour, hierarchy = cv2.findContours(img_array[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.append(contour)

    for i in range(0, 3):
        for cnt in contours[i]:
            area = cv2.contourArea(cnt)
            if area == 89627.5 or area == 107568.5 or area == 1372597.0:  # outer borders without dilation
                cv2.drawContours(img_array[i], cnt, -1, (0, 0, 0), 5)
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
                img_cropped = cv2.warpPerspective(img_array[i], matrix, (width, height))

                # binary image
                _, img_cropped = cv2.threshold(img_cropped, 125, 255, cv2.THRESH_BINARY_INV)
                plt.imshow(img_cropped, cmap="gray")
                plt.show()

                # crop cells
                for y in range(1, 10):
                    for x in range(1, 10):
                        cell = img_cropped[y * 100 - 100:y * 100, x * 100 - 100:x * 100]
                        # removing borders
                        for cell_x in range(0, 100):
                            for cell_y in range(0, 100):
                                if (cell_x < 15 or cell_x > 85) or (cell_y < 15 or cell_y > 85):
                                    cell[cell_x][cell_y] = 255
                        images_array.append(cell)


def time_and_accuracy():
    global images_array
    numbers_array = []
    times = []
    grids = []

    # time measure
    for i in range(0, amount_of_images):
        start = time.time()
        value = pytesseract.image_to_string(images_array[i], config='--psm 6')  # --psm 6 - assume a single uniform block of text.
        end = time.time()
        read_time = end - start
        times.append(read_time)
        if not value:
            value = 0
        else:
            value = value[0]  # output before: i.e. '7\n'
        numbers_array.append(value)
        if i == 80 or i == 161 or i == 242:
            grids.append(np.reshape(numbers_array[i-80:i+1], (9, 9)))

    # comparing values
    read_values = np.array(grids).flatten()
    expected_values = np.array(grid1 + grid2 + grid3).flatten()
    accuracy_array_zero = []
    accuracy_array_nonzero = []
    for i in range(0, amount_of_images):
        if expected_values[i] != '0':
            if read_values[i] == expected_values[i]:
                accuracy_array_nonzero.append(1)
            else:
                accuracy_array_nonzero.append(0)
        else:
            if read_values[i] == expected_values[i]:
                accuracy_array_zero.append(1)
            else:
                accuracy_array_zero.append(0)
    print("--------------------------------------------------------")
    print("Average pytesseract.image_to_string read time: {:.4f}".format(np.mean(times)), "s.")
    print("\nOnly digit cells:")
    print("Amount of images:", len(accuracy_array_nonzero), ".")
    print("Accuracy: {:.2f}".format((sum(accuracy_array_nonzero) / len(accuracy_array_nonzero)) * 100), "%.")
    print("\nAll cells:")
    print("Amount of images:", (len(accuracy_array_zero)+len(accuracy_array_nonzero)), ".")
    print("Accuracy: {:.2f}".format((sum(accuracy_array_zero+accuracy_array_nonzero) / (len(accuracy_array_zero)+len(accuracy_array_nonzero))) * 100), "%.")
    print("--------------------------------------------------------")


def main():
    print("\nProgram is running...")
    get_images_array()
    time_and_accuracy()


# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()
