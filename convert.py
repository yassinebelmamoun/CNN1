import numpy as np
from scipy.misc import toimage, imsave
import os

in_dir = "imgs_mask_test"

images_to_convert_path = './' + in_dir + '.npy'

img_array = np.load(images_to_convert_path)

output_dir = 'binary_image'

list_of_random_index = range(9) 

def ensure_directory_exist(directory_name):
    exist_bool = os.path.isdir('./' + directory_name)
    if not exist_bool:
        os.mkdir(directory_name)

def show_image(image):
    toimage(image).show()

def plot_image_save_to_file(name, img_cur):
    save_directory = output_dir  # from global value
    ensure_directory_exist(save_directory)
    file_name = name + '.png'
    full_path = os.path.join(save_directory, file_name)
    imsave(full_path, img_cur)

def convert_numpy_array_to_int_array(img_array):
    print(len(img_array))   # will return number of pictures
    image_list = []
    i = 0
    while i < len(img_array):
        for photo_indiv in img_array[i]:
            mean = photo_indiv.mean()
            """
                HERE IS THE FILTER
            """
            photo_indiv = np.where(photo_indiv > 0.5, 1, 0)
            image = photo_indiv.astype('float32')
            image_list.append(image)
        i += 1
    return image_list

def convert_int_array_to_png(image_list):
    ind_id = 1
    for photo_array in image_list:
        name = in_dir + '_' + str(ind_id)
        plot_image_save_to_file(name, photo_array)
        ind_id += 1

def get_random_5(img_array_int):
    mySet = set()
    smaller_list = []

    for selected_index in list_of_random_index:
        mySet.add(selected_index)

    i = 0
    while i < len(img_array_int):
        if i in mySet:
            smaller_list.append(img_array_int[i])
        i += 1

    return smaller_list

def convert_random_5(img_array_int):
    smaller_list = get_random_5(img_array_int)
    convert_int_array_to_png(smaller_list)

def main():
    img_array_int = convert_numpy_array_to_int_array(img_array)
    convert_random_5(img_array_int)

main()
