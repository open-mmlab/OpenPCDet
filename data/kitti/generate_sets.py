import random
import sys

default_size = 7518
percentage_cut = float(sys.argv[1])
num_frames = int(default_size * percentage_cut)


data = list(range(num_frames))
random.shuffle(data)

train_data = sorted(data[int(num_frames/2) :])
val_data = sorted(data[: int(num_frames/2)])

with open("ImageSets/train.txt", 'w') as file:
    for el in train_data:
        file.write('%06d\n' % el)

with open("ImageSets/val.txt", 'w') as file:
    for el in val_data:
        file.write('%06d\n' % el)

with open("ImageSets/test.txt", 'w') as file:
    for el in range(default_size):
        file.write('%06d\n' % el)