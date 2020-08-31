import numpy as np
from scipy.misc import imread,imsave


num_to_read = 90000
save_path = '/local/sdb/wd263/my_dsprites/'
image_name_format = save_path + 'raw_occluded_hh/img_{}.png'

img_array = []

for i in range(num_to_read):
    img = imread(image_name_format.format(i),mode='L')
    img_array.append(img)

labels = np.load(save_path + 'raw_occluded_hh/labels.npz')['labels']
img_array = np.array(img_array)

to_save = np.random.randint(num_to_read)
print(to_save)
imsave('test_save.png',img_array[to_save])
print(img_array.shape)
np.savez_compressed(save_path+'dsprites_data_ochh',images = img_array, labels = labels)
