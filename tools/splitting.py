import matplotlib.pyplot
import numpy as np

matplotlib.use('TkAgg')


# split image and label somewhere in the middle
def split_image_and_label(image, label, settings):
    middle = find_empty_middle(image)
    image_left, image_right = split_array(image, middle)
    label_left, label_right = split_array(label, middle)

    return image_left, image_right, label_left, label_right


# split image (without label) somewhere in the middle
def split_image(image):
    toleranz = 0
    middle = 0
    while middle == 0:
        toleranz += 50
        middle = find_empty_middle(image, toleranz)
    image_left, image_right = split_array(image, middle)
    return image_left, image_right


# split an array in the given middle
def split_array(array, middle):
    array_left = array[0:middle, :, :]
    array_right = array[middle:array.shape[0], :, :]
    return array_left, array_right


# split an image somewhere in the middle
def find_empty_middle(array, tolerance=50):
    mask = array > tolerance
    m, n, b = mask.shape

    middle_start = int(n / 2 - 30)
    middle_list = []
    mean = 100
    max = 500
    middle_not_really = 0

    for x in range(middle_start, middle_start + 60):
        if np.all(mask[x, :, :] == False):
            middle_list.append(int(x))
        else:
            new_mean = np.mean(array[x, :, :])
            new_max = np.max(array[x, :, :])

            if new_mean < mean and new_max < max:
                mean = new_mean
                max = new_max
                middle_not_really = int(x)

    if not middle_list:
        middle_middle = middle_not_really
    else:
        middle_middle = int(np.mean(middle_list))

    return middle_middle
