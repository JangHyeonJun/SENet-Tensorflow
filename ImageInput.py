import os
import sys
import time
import pickle
import random
import csv
from PIL import Image
import numpy as np

class_num = 6
image_size = 128
img_channels = 3

def download_data():
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = 'cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet aready exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif (fname.endswith("tar")):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

def prepare_data():
    print("======Loading data======")
    data_dir = './SENet/faces_images'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels

def load_data(csv_path, file_num):
    with open(csv_path) as csvfile:
        rdr = csv.reader(csvfile)
        batch_one_hot_label = np.zeros((file_num, class_num))
        batch_image = np.zeros((file_num, image_size, image_size, img_channels))
        index = -1
        for i in rdr:
            if i[0] != 'filename':
                index += 1
                filename = i[0]
                label_num = int(i[1]) - 1
                dir_path = './SENet/faces_images'
                file_path = dir_path + '/' + filename
                batch_one_hot_label[index, :] = np.eye(class_num)[label_num]
                batch_image[index, :, :, :] = np.array(Image.open(file_path))

        indices = np.random.permutation(file_num)
        slice_index = int(len(indices) * 0.9)
        train_indices = indices[:slice_index]
        test_indices = indices[slice_index:]

        train_data = batch_image[train_indices]
        train_label = batch_one_hot_label[train_indices]
        test_data = batch_image[test_indices]
        test_label = batch_one_hot_label[test_indices]

    return train_data, train_label, test_data, test_label

train_x, train_y, test_x, test_y = load_data('./SENet/train_vision.csv', 5850)
print(np.shape(train_x))