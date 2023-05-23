import numpy as np
from scipy import ndimage


def crop_image_and_label_background(image, label):
    col_start, col_end, row_start, row_end = find_core_image(image)
    cropped_image = crop_array_in_two_dimensions(image, col_start, col_end, row_start, row_end)
    cropped_label = crop_array_in_two_dimensions(label, col_start, col_end, row_start, row_end)
    return cropped_image, cropped_label


# cops all the empty columns and rows in image (without label)
def crop_image_background(image):
    col_start, col_end, row_start, row_end = find_core_image(image)
    temp = [col_start, col_end, row_start, row_end]
    cropped_image = crop_array_in_two_dimensions(image, col_start, col_end, row_start, row_end)
    return cropped_image, temp


# crops the background of an array with
def crop_array_in_two_dimensions(array, col_start, col_end, row_start, row_end):
    return array[col_start:col_end, row_start:row_end, :]


# find where image-value is higher than tolerance (-> where there is not only background in column/row)
def find_core_image(image):
    tolerance = 0
    boolean_mask = image == tolerance
    m, n, b = boolean_mask.shape
    # 150x320x25 -> 150x320 merged into one slide
    conv_into_2d = np.all(boolean_mask, axis=2)
    # merge into high 320
    first_dim_mask = np.all(conv_into_2d, axis=0)
    # merge into width 150
    second_dim_mask = np.all(conv_into_2d, axis=1)

    # length 320
    row_start = first_dim_mask.argmin()
    row_end = n - first_dim_mask[::-1].argmin()
    # width 150
    col_start = second_dim_mask.argmin()
    col_end = m - second_dim_mask[::-1].argmin()

    return col_start, col_end, row_start, row_end


def resize_list_img_and_label_to_given_dim(list_img, list_label, input):
    resized_list_img, factor = resize_list_of_arrays_to_specific_dimension(list_img, input, False)
    resized_list_label, factor = resize_list_of_arrays_to_specific_dimension(list_label, input, True)
    return resized_list_img, resized_list_label


def resize_list_img_to_given_dim(list_img, input_size):
    resized_list_img, factor = resize_list_of_arrays_to_specific_dimension(list_img, input_size, False)
    return resized_list_img, factor


# make list of arrays even, by adding zeroes to fit a given dimension
def resize_list_of_arrays_to_specific_dimension(list_arrays, input, label):
    resized_array_list = []
    # print(list_arrays[0].shape)
    scaling_factor = []

    for array in list_arrays:
        scaling_factor_each = []
        if label:
            resized_image = np.zeros(input.inputShape_create, dtype=np.uint8)
        else:
            resized_image = np.zeros(input.inputShape_create)

        x_factor = input.inputShape_create[0] / array.shape[0]
        y_factor = input.inputShape_create[1] / array.shape[1]
        z_factor = input.inputShape_create[2] / array.shape[2]
        minimal_factor = min(x_factor, y_factor)
        array = ndimage.zoom(array, zoom=[minimal_factor, minimal_factor, z_factor], order=0)
        scaling_factor_each.append(minimal_factor)
        scaling_factor_each.append(z_factor)
        if array.shape[2] < input.inputShape_create[2]:
            resized_image[:array.shape[0], :array.shape[1], :array.shape[2]] = array
        else:
            resized_image[:array.shape[0], :array.shape[1], :array.shape[2]] = array[:, :, :input.inputShape_create[2]]
        scaling_factor_each.append(array.shape)
        resized_array_list.append(resized_image)
        scaling_factor.append(scaling_factor_each)

    return np.array(resized_array_list), scaling_factor
