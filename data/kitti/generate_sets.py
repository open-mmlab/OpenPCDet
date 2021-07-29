import sys

default_size = 7481
percentage_cut = float(sys.argv[1])
num_frames = int(default_size * percentage_cut)

data = list(range(num_frames))

val_data = []
with open("ImageSets/val.txt", 'r') as file:
    for el in file:
        val_data.append(int(el))

train_size = 0
with open("ImageSets/train.txt", 'w') as file:
    for el in data:
        if el not in val_data and train_size < default_size/2:
            file.write('%06d\n' % el)
            train_size += 1

print(f"Generated {train_size} training id's")

'''
with open("ImageSets/val.txt", 'w') as file:
    for el in val_data:
        file.write('%06d\n' % el)

print(f"Generated {len(val_data)} validation id's")


with open("ImageSets/test.txt", 'w') as file:
    for el in range(default_size):
        file.write('%06d\n' % el)

print(f"Generated {default_size} testing id's")
'''
