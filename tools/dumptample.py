import os
import sys
import time
import copy
import json
import numpy as np

def dump_calib_tables(fname):
    try:
        with open(fname, 'r') as handle:
            calib_dict = json.load(handle)
    except FileNotFoundError:
        print(f'Calibration file {fname} not found')
        return
    # Use 99 percentile Post-PFE times
    def get_rc(k):
        r,c = k.replace('(', '').replace(')', '').replace(',', '').split()
        r,c = int(r), int(c)
        return r,c

    post_sync_time_table_ms = [
        # heads    1    2    3    4    5    6
        [ 10., 20., 30., 40., 50., 60.], # no more rpn stage
        [ 25., 35., 45., 55., 65., 75.], # 1  more rpn stage
        [ 40., 50., 60., 70., 80., 90.], # 2  more rpn stages
    ]
    
    cfg_to_NDS = [
        [.2] * 6,
        [.3] * 6,
        [.4] * 6,
    ]

    for k, v in calib_dict['stats'].items():
        r,c = get_rc(k)
        post_sync_time_table_ms[r-1][c-1] = v['Post-PFE'][3]

    for k, v in calib_dict['eval'].items():
        r,c = get_rc(k)
        cfg_to_NDS[r-1][c-1] = round(v['NDS'], 3)

    tuples=[]
    for i in range(len(post_sync_time_table_ms)):
        for j in range(len(post_sync_time_table_ms[0])):
            tuples.append((post_sync_time_table_ms[i][j],cfg_to_NDS[i][j], i, j))

    tuples.sort(key=lambda x: -x[1])
    t = 0
    while t < len(tuples)-1:
       if tuples[t][0] < tuples[t+1][0]:
           tuples.pop(t+1)
       else:
            t += 1

    for t in tuples:
        print(t)

#        for i in range(len(self.cfg_to_NDS)):
#            for j in range(1, len(self.cfg_to_NDS[0])):
#                if self.cfg_to_NDS[i][j-1] >= self.cfg_to_NDS[i][j]:
#                    self.cfg_to_NDS[i][j] = self.cfg_to_NDS[i][j-1] + 0.001

    print('Post PFE wcet table:')
    for row in post_sync_time_table_ms:
        row = [str(r) for r in row]
        print('\t'.join(row))

    print('Stage/Head configuration to NDS table:')
    for row in cfg_to_NDS:
        row = [str(r) for r in row]
        print('\t'.join(row))

dump_calib_tables(sys.argv[1])
