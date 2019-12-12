from PIL import Image
import csv
import numpy as np

def preprocess_image(csv_path):
    with open(csv_path) as csvfile:
        rdr = csv.reader(csvfile)
        index = -1
        for i in rdr:
            if i[0] != 'filename':
                filename = i[0]
                dir_path = './SENet/faces_images'
                save_dir_path = './SENet/PPImage'
                file_path = dir_path + '/' + filename
                save_file_path = save_dir_path + '/' + filename
                origin = Image.open(file_path)
                resize = origin.resize((32,32))
                resize.save(save_file_path)

preprocess_image('./SENet/train_vision.csv')
print('전처리 끗')