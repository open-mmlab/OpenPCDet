import sys
import os
import numpy as np

def calculate_anchors(data_dir):
    class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']
    # class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown']
    res = {}
    
    train_list = os.path.join(data_dir, 'ImageSets', 'train.txt')
    val_list = os.path.join(data_dir, 'ImageSets', 'val.txt')
    label_dir = os.path.join(data_dir, 'training', 'label_2')

    train_file = open(train_list, 'r')
    train_lines = train_file.readlines()
    train_file.close()

    for line in train_lines:
        path = os.path.join(label_dir, line.strip() + '.txt')
        f = open(path, 'r')
        labels = f.readlines()
        for label in labels:
            class_name = label.split(' ')[0]
            h = label.split(' ')[8]
            w = label.split(' ')[9]
            l = label.split(' ')[10]
            hwl = np.array([float(h), float(w), float(l)])
            if class_name in class_names:
                try:
                    res[class_name] = np.insert(res[class_name], -1, values=hwl, axis=0)
                except KeyError:
                    res[class_name] = np.array([hwl])

    print('train set anchor size h, w, l:')
    for class_name, hwls in res.items():
        print('%s: ' % class_name)
        print(np.mean(hwls, axis=0))


if __name__ == '__main__':
    data_dir = sys.argv[1]
    calculate_anchors(data_dir)