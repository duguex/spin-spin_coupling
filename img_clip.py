import cv2
import numpy as np


# find contours in the image
def find_contours(thresh, index_range):
    for index in index_range:
        if not np.all(thresh[index] == 255):
            return index


def clip_backgroud(img_path_list, bg_witdh=10):
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    img_list = []
    for img_path in img_path_list:
        # read the image
        img = cv2.imread(img_path)
        img_list.append(img)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # threshold the image to get the white background
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        y_min.append(find_contours(thresh, range(thresh.shape[0])))
        y_max.append(find_contours(thresh, range(thresh.shape[0])[::-1]))
        x_min.append(find_contours(thresh.T, range(thresh.T.shape[0])))
        x_max.append(find_contours(thresh.T, range(thresh.T.shape[0])[::-1]))

    x_range = [min(x_min), max(x_max)]
    y_range = [min(y_min), max(y_max)]

    for img, img_path in zip(img_list, img_path_list):
        cv2.rectangle(img, (x_range[0] - bg_witdh, y_range[0] - bg_witdh),
                      (x_range[1] + bg_witdh, y_range[1] + bg_witdh), (0, 255, 0), 2)
        # show the image
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(img_path.replace(".", "_clip."), img[y_range[0] - bg_witdh:y_range[1] + bg_witdh,
                                                     x_range[0] - bg_witdh:x_range[1] + bg_witdh])


if __name__ == '__main__':
    work_dir = r"C:\Users\dugue\Desktop"
    img_list = [work_dir + f'/Figure_{i}.png' for i in range(1, 7)]
    clip_backgroud(img_list)
