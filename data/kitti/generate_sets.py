import random
import os, os.path
import sys

num_frames = int(sys.argv[1])
dataset_size = len(os.listdir('training/calib'))

data = list(range(num_frames))
random.shuffle(data)

train_data = sorted(data[int(num_frames/2) :])
val_data = sorted(data[: int(num_frames/2)])

with open("ImageSets/test/train.txt", 'w') as file:
    for el in train_data:
        file.write('%06d\n' % el)

with open("ImageSets/test/val.txt", 'w') as file:
    for el in val_data:
        file.write('%06d\n' % el)

with open("ImageSets/test/test.txt", 'w') as file:
    for el in range(dataset_size):
        file.write('%06d\n' % el)