import os 

velDir = '/home/yagmur/OpenPCDet/data/lyft/training/velodyne'

trainfile = open('train.txt', 'w+')

for filename in  os.listdir(velDir):
    filename = filename.split('.')[0]
    trainfile.write(filename + '\n')
    
trainfile.close()    

