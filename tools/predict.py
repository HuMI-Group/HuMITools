import numpy as np
from scipy import ndimage

from tools import splitting, cropping_and_resizing


def preprocess_before_predict(image_array_raw, settings):
    temp_left_all = []
    temp_right_all = []
    temp_list = []
    images_train_list = []
    after_split_shape_all = []

    rounded_times_split = image_array_raw.shape[2] // settings.inputShape_create[2]
    if rounded_times_split != 0:
        list_splittet_image_arrays = np.array_split(image_array_raw, rounded_times_split, axis=2)
    else:
        list_splittet_image_arrays = [image_array_raw]

    index = 0
    overlapping_list_image = []
    for part_image in list_splittet_image_arrays:
        if index > 0:
            part_image = np.concatenate((list_splittet_image_arrays[index - 1][:, :, -3:], part_image),
                                        axis=2)
        if index < (len(list_splittet_image_arrays) - 1):
            part_image = np.concatenate((part_image, list_splittet_image_arrays[index + 1][:, :, :3]), axis=2)

        overlapping_list_image.append(part_image)
        index += 1
    for image_array in overlapping_list_image:
        image_array *= 100.0 / image_array.max()
        # split
        if settings.split:
            image_array_left, image_array_right = splitting.split_image(image_array)
            after_split_shape = (image_array_left.shape, image_array_right.shape)
            after_split_shape_all.append(after_split_shape)
            processed_image_left, temp_left = cropping_and_resizing.crop_image_background(image_array_left)
            processed_image_right, temp_right = cropping_and_resizing.crop_image_background(image_array_right)
            images_train_list.append(processed_image_left)
            images_train_list.append(processed_image_right[::-1, :, :])
            temp_left_all.append(temp_left)
            temp_right_all.append(temp_right)
        else:
            after_split_shape = image_array.shape
            after_split_shape_all.append(after_split_shape)
            processed_image, temp = cropping_and_resizing.crop_image_background(image_array)
            images_train_list.append(processed_image)
            temp_list.append(temp)
    resized_images_list, scaling_factor_all = cropping_and_resizing.resize_list_img_to_given_dim(
        images_train_list, settings)

    resized_images_list_pre = resized_images_list[..., np.newaxis]
    return after_split_shape_all, resized_images_list_pre, scaling_factor_all, temp_left_all, temp_list, temp_right_all


def merge_overlappings(new_label_list, new_label_pred_list, probabilty_label_list):
    index = 0

    for part_probability in probabilty_label_list:
        part = np.floor(part_probability)
        prop_only = part_probability - part
        if index > 0:
            part = part[:, :, 6:]
            part_probability = part_probability[:, :, 6:]

        if index < (len(probabilty_label_list) - 1):
            first = part[:, :, -6:]
            second = np.floor(probabilty_label_list[index + 1])[:, :, :6]

            combined = np.zeros(first.shape)
            combined_prob = np.zeros(first.shape)

            first_prob = prop_only[:, :, -6:]
            second_prob = probabilty_label_list[index + 1][:, :, :6]

            f_ranges = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
            s_ranges = [0.1, 0.2, 0.4, 0.6, 0.8, 1]

            first_counter = 0
            second_counter = 5
            for f_range, s_range in zip(f_ranges, s_ranges):
                first_prob[:, :, first_counter] *= f_range
                second_prob[:, :, second_counter] *= s_range
                first_counter += 1
                second_counter -= 1

            combined[first_prob > second_prob] = first[first_prob > second_prob]
            combined[second_prob > first_prob] = second[second_prob > first_prob]
            part[:, :, -6:] = combined

            combined_prob[first_prob > second_prob] = first_prob[first_prob > second_prob]
            combined_prob[second_prob > first_prob] = first_prob[second_prob > first_prob]
            part_probability[:, :, -6:] = combined_prob

        new_label_list.append(part)
        new_label_pred_list.append(part_probability)
        index += 1


def undo_reshaping_stuff(after_split_shape, list_to_revert, scaling_factor_all,
                         input):
    list_scaled_labels = []
    for to_revert in list_to_revert:
        scaling_factor = scaling_factor_all[to_revert[0]]
        scaled_label = to_revert[1][:scaling_factor[2][0], :scaling_factor[2][1],
                       :scaling_factor[2][2]]
        z_scaling_l = 1 / scaling_factor[1]
        scaled_label = re_rescale(to_revert[2], z_scaling_l, scaled_label)
        list_scaled_labels.append(scaled_label)

    if input.split:
        scaled_right_label = list_scaled_labels[1][::-1, :, :]
        scaled_left_label = list_scaled_labels[0]
        temp_left = list_to_revert[0][2]
        temp_right = list_to_revert[1][2]

        scaled_left_label, scaled_right_label = rename_labels(scaled_left_label, scaled_right_label, input)
        final_output_left = np.zeros(after_split_shape[0])
        final_output_right = np.zeros(after_split_shape[1])

        if (final_output_left[temp_left[0]:temp_left[1], temp_left[2]:temp_left[3], :].shape == scaled_left_label.shape) \
                and (final_output_right[temp_right[0]:temp_right[1], temp_right[2]:temp_right[3],
                     :].shape == scaled_right_label.shape):
            final_output_left[temp_left[0]:temp_left[1]:, temp_left[2]:temp_left[3]:, :] = scaled_left_label
            final_output_right[temp_right[0]:temp_right[1], temp_right[2]:temp_right[3], :] = scaled_right_label
        else:
            final_output_left[:scaled_left_label.shape[0], :scaled_left_label.shape[1], :] = scaled_left_label
            final_output_right[:scaled_right_label.shape[0], :scaled_right_label.shape[1], :] = scaled_right_label
            print('errooooorr')
            print('###############')
        return np.concatenate((final_output_left, final_output_right), axis=0)
    else:
        final_output = np.zeros(after_split_shape)
        temp = list_to_revert[0][2]

        if final_output[temp[0]:temp[1], temp[2]:temp[3], :].shape == list_scaled_labels[0].shape:
            final_output[temp[0]:temp[1]:, temp[2]:temp[3]:, :] = list_scaled_labels[0]
        else:
            final_output[:list_scaled_labels[0].shape[0], :list_scaled_labels[0].shape[1], :] = list_scaled_labels[0]
            print('errooooorr')
            print('###############')
        return final_output


def rename_labels(left_side, right_side, settings):
    correct_values_left = settings.labels_left
    correct_values_right = settings.labels_right

    # create empty
    new_left = np.zeros(left_side.shape)
    new_right = np.zeros(right_side.shape)

    index = 0
    for left_val, right_val in zip(correct_values_left, correct_values_right):
        new_left[left_side == index] = left_val
        new_right[right_side == index] = right_val
        index += 1
    return new_left, new_right


def re_rescale(temp_size, zscaling, array):
    size_x = temp_size[1] - temp_size[0]
    size_y = temp_size[3] - temp_size[2]

    scaling_x = size_x / array.shape[0]
    scaling_y = size_y / array.shape[1]
    scaled_label = ndimage.zoom(array, zoom=[scaling_x, scaling_y, zscaling], order=0)
    return scaled_label
