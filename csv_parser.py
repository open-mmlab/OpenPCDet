import sys
import os

def main():
    filepath = sys.argv[1]

    if not os.path.isdir(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()
       
    file_list = os.listdir(filepath)
    file_list = sorted(list(filter(lambda k: ('log_train' in k or 'log_eval') and ('details' not in k), file_list)))

    csv_line = ""
    for log in file_list:
        if ".txt" in log:
            flag = False
            with open(filepath + '/' + log) as f:
                for line in f:
                    if 'Performance of EPOCH' in line:
                        csv_line = csv_line + line.split("EPOCH ",1)[1].split(" **")[0] + ', '
                    elif ('Car AP_R40@0.70, 0.70, 0.70:' in line) or ('Pedestrian AP_R40@0.50, 0.50, 0.50:' in line):
                        flag = True
                    elif '3d' in line and flag:
                        csv_line = csv_line + line.split(":",1)[1]
                        flag = False
    if csv_line != "":
        with open(filepath + '/eval.csv', 'w') as csv:
            csv.write(csv_line)

if __name__ == '__main__':
    main()