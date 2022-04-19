import os, re
import glob
import sys
import copy
import json
import math
import gc
import threading
import concurrent.futures
from multiprocessing import Process
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Default evaluation dict format
# All values are 1D_LISTs
proto_exec_time_dict = {
    'End-to-end': [],
#    'PreProcess': [],
#    'VFE': [],
#    'MapToBEV': [],
#    'AnchorMask': [],
#    'RPN-stage-1': [],
#    'RPN-stage-2': [],
#    'RPN-stage-3': [],
#    'RPN-finalize': [],
#    'RPN-total': [],
#    'Pre-stage-1': [],
#    'Post-stage-1': [],
#    'PostProcess': [],
}

proto_AP_types_dict = None
proto_AP_dict = None
proto_mAP_dict = None
proto_eval_dict = None
dataset = 'NuScenes'

def init_dicts(dataset_name):
    global proto_AP_types_dict
    global proto_AP_dict
    global proto_mAP_dict
    global proto_eval_dict
    global dataset

    if dataset_name == 'KittiDataset':
        proto_AP_types_dict = {
            "aos": [],
            "3d": [],
            "bev": [],
            "image": [],
        }

        # Rows will be aos image bev 3d, cols will be easy medium hard
        proto_AP_dict = {
            'Car': copy.deepcopy(proto_AP_types_dict),
            'Pedestrian': copy.deepcopy(proto_AP_types_dict),
            'Cyclist': copy.deepcopy(proto_AP_types_dict),
        }

        proto_mAP_dict = {
            "aos":0.0,
            '3d': 0.0,
            'bev': 0.0,
            'image': 0.0,
        }

    elif dataset_name == 'NuScenesDataset':
        proto_AP_types_dict = {
            "AP": [], # 0.5, 1.0, 2.0, 4.0
        }

        proto_AP_dict = {
            'car': copy.deepcopy(proto_AP_types_dict),
            'pedestrian': copy.deepcopy(proto_AP_types_dict),
            'traffic_cone': copy.deepcopy(proto_AP_types_dict),
            'motorcycle': copy.deepcopy(proto_AP_types_dict),
            'bicycle': copy.deepcopy(proto_AP_types_dict),
            'bus': copy.deepcopy(proto_AP_types_dict),
            'trailer': copy.deepcopy(proto_AP_types_dict),
            'truck': copy.deepcopy(proto_AP_types_dict),
            'construction_vehicle': copy.deepcopy(proto_AP_types_dict),
            'barrier': copy.deepcopy(proto_AP_types_dict),
        }

        proto_mAP_dict = {
            'NDS': 0.0,
            'mAP': 0.0,
        }
    else:
        print('Unknown dataset')
        return
    dataset = dataset_name

    proto_eval_dict = {
        'method': 1,  # VAL
        'rpn_stg_exec_seqs': [],
        'gt_counts': [],  # 2D_LIST
        'deadline_sec': 0.1,  # VAL
        'deadline_msec': 100.,  # VAL
        'deadlines_missed': 0,  # VAL
        'deadline_diffs': [],  # 1D_LIST
        'exec_times': proto_exec_time_dict,  # DICT
        'exec_time_stats': proto_exec_time_dict,  # DICT
        "AP": proto_AP_dict,
        "mAP": proto_mAP_dict,
        'eval_results_dict': {},
        "dataset": 'NuScenesDataset',
        "time_err": {},
        'avrg_recognize_time': 0.,
        'avrg_instances_detected': 0.,
        'color': 'r',
        'lnstyle': '-',
    }

# method number to method name
# NANP: No aging no prediction


#linestyles = ['-', '--', '-.', ':'] * 4
linestyles = ['--', ':'] * 10
method_colors= [
    'tab:blue', 
    'tab:orange', 
    'tab:green', 
    'tab:olive', 
    'tab:purple', 
    'w',
    'tab:brown',
    'tab:blue', 
    'w',
    'tab:red',  #'xkcd:coral', 
    'tab:pink', 
    'tab:orange', 
    'tab:green'
]
m_to_c_ls = [(method_colors[i], linestyles[i]) for i in range(len(method_colors))]

method_num_to_str = [
        '3Baseline-1',
        '2Baseline-2',
        '1Baseline-3',
        '4MultiStage',
        '5RoundRoHS',
        'PCHeadSel',
        '6HistoryHS',
        '8RoundRoHS-P',
        'PCHeadSel-P',
        'AHistoryHS-P',
        '7StaticHS',
        '9CSSumHS-P',
        'BNearOptHS-P',
]

def merge_eval_dicts(eval_dicts):
    merged_ed = copy.deepcopy(proto_eval_dict)

    proto_keys_set = set(proto_eval_dict.keys())
    for ed in eval_dicts:
        missing_keys_set = set(ed.keys())
        diff_keys = proto_keys_set.difference(missing_keys_set)
        for k in diff_keys:
            ed[k] = proto_eval_dict[k]
            
    # use np.concatenate on a 2D list if you want to merge multiple 2D arrays
    for k, v in merged_ed.items():
        if not isinstance(v, dict):
            merged_ed[k] = [e[k] for e in eval_dicts]

    for k1 in ['exec_times', 'exec_time_stats',  'mAP']:
        for k2 in proto_eval_dict[k1].keys():
            merged_ed[k1][k2] = [e[k1][k2] \
                    for e in eval_dicts]
#                    for e in eval_dicts if e[k1].__contains__(k2)]

    for cls in merged_ed['AP'].keys():
        for eval_type in proto_AP_dict[cls].keys():
            merged_ed['AP'][cls][eval_type] = \
                [e['AP'][cls][eval_type] for e in eval_dicts]

    return merged_ed

inp_dir = sys.argv[1]

for path in glob.glob(inp_dir + "/eval_dict_*"):
    with open(path, 'r') as handle:
        eval_d = json.load(handle)
        init_dicts(eval_d.get('dataset','KittiDataset'))
    break

# each experiment has multiple eval dicts
def load_eval_dict(path):
    print('Loading', path)
    with open(path, 'r') as handle:
        eval_d = json.load(handle)
    eval_d['deadline_msec'] = int(eval_d['deadline_sec'] * 1000)

    dataset = eval_d.get('dataset','KittiDataset')
    # Copy AP dict with removing threshold info, like @0.70
    AP_dict_json = eval_d["eval_results_dict"]
    AP_dict = copy.deepcopy(proto_AP_dict)
    if dataset == 'KittiDataset':
        for cls_metric, AP in AP_dict_json.items():
            cls_metric, difficulty = cls_metric.split('/')
            if cls_metric == 'recall':
                continue
            cls, metric = cls_metric.split('_')
            AP_dict[cls][metric].append(AP)
        for v in AP_dict.values():
            for v2 in v.values():
                v2.sort() # sort according to difficulty

        eval_d["AP"] = AP_dict

        # Calculate mAP values
        eval_d["mAP"] = copy.deepcopy(proto_mAP_dict)
        for metric in eval_d["mAP"].keys():
            mAP, cnt = 0.0, 0
            for v in eval_d["AP"].values():
                mAP += sum(v[metric])  # hard medium easy
                cnt += len(v[metric])  # 3
            if cnt > 0:
                eval_d["mAP"][metric] = mAP / cnt
    elif dataset == 'NuScenesDataset':
        results = AP_dict_json['result_str'].split('\n')
        for i, r in enumerate(results):
            for cls in proto_AP_dict.keys():
                if cls in r:
                    AP_scores = results[i+1].split('|')[1].split(',')
                    AP_scores = [float(a.strip()) for a in AP_scores]
                    AP_dict[cls]['AP'] = AP_scores

        eval_d["AP"] = AP_dict
        # Get mAP values
        eval_d["mAP"] = copy.deepcopy(proto_mAP_dict)
        eval_d['mAP']['NDS'] = AP_dict_json['NDS']
        eval_d['mAP']['mAP'] = AP_dict_json['mAP']

        if 'time_err' in eval_d:
            eval_d['avrg_recognize_time'] = 0.
            eval_d['avrg_instances_detected'] = 0
            for cls, timings_per_thr in  eval_d['time_err'].items():
                avrg_timing, avrg_instances = .0, .0
                for timings in timings_per_thr:
                    l = len(timings)
                    if l > 0:
                        avrg_timing += sum(list(timings.values())) / l
                        avrg_instances += l
                eval_d['avrg_recognize_time'] += avrg_timing / len(timings_per_thr)
                eval_d['avrg_instances_detected'] += avrg_instances/ len(timings_per_thr)
            eval_d['avrg_recognize_time'] /= len(eval_d['time_err'])
            eval_d['avrg_instances_detected'] /= len(eval_d['time_err'])

    return eval_d

exps_dict = {}
# load eval dicts
with concurrent.futures.ThreadPoolExecutor() as executor:
    futs = []
    paths = sorted(glob.glob(inp_dir + "/eval_dict_*"))
    for path in paths:
        futs.append(executor.submit(load_eval_dict, path))
    for f in concurrent.futures.as_completed(futs):
        ed = f.result()
        k = method_num_to_str[ed['method']-1]
        if k not in exps_dict:
            exps_dict[k] = []
        ed['color'] = m_to_c_ls[ed['method']-1][0]
        ed['lnstyle'] = m_to_c_ls[ed['method']-1][1]
        exps_dict[k].append(ed)

#Sort exps
exp_names = sorted(exps_dict.keys())
exps_dict= {nm[1:]:exps_dict[nm] for nm in exp_names}

# Filter some
#exps_dict = {nm:exps_dict[nm] for nm in ['Baseline-3', 'Baseline-2', 'Baseline-1', 'Impr-MS-HS-A-P']}

max_NDS = 0.
for exp_name, evals in exps_dict.items():
    NDS_arr = [e['mAP']['NDS'] for e in evals]
    max_NDS = max(max(NDS_arr), max_NDS)

exps_dict1= { nm:exps_dict[nm] for nm in [ \
        'Baseline-1',
        'Baseline-2',
        'Baseline-3',
        'MultiStage',
        'HistoryHS-P',
]}

exps_dict2 = { nm:exps_dict[nm] for nm in [ \
        'MultiStage',
        'StaticHS',
        'RoundRoHS',
        'HistoryHS',
        'HistoryHS-P',
]}

exps_dict3 = { nm:exps_dict[nm] for nm in [ \
        'CSSumHS-P',
        'RoundRoHS-P',
        'HistoryHS-P',
        'NearOptHS-P',
]}

exps_dict=exps_dict3

plot_head_selection = True

for exp, evals in exps_dict.items():
    # Sort according to deadlines
    evals.sort(key=lambda e: e['deadline_sec'])
    evals.sort(key=lambda e: e['deadline_msec'])
    print('Experiment:',exp)
    for e in evals:
        if dataset == 'KittiDataset':
            mAP_image, mAP_bev, mAP_3d = e["mAP"]['image'], e["mAP"]['bev'], e["mAP"]['3d']
            print('\tdeadline:', e['deadline_sec'], "\tmissed:", e['deadlines_missed'],
                  f"\tmAP (image, bev, 3d):\t{mAP_image:.2f},\t{mAP_bev:.2f},\t{mAP_3d:.2f}")
        elif dataset == 'NuScenesDataset':
            mAP, NDS = e["mAP"]['mAP'], e["mAP"]['NDS']
            print('\tdeadline:', e['deadline_sec'], "\tmissed:", e['deadlines_missed'],
                  f"\tmAP, NDS:\t{mAP:.2f},\t{NDS:.2f}")
merged_exps_dict = {}
for k, v in exps_dict.items():
    merged_exps_dict[k] = merge_eval_dicts(v)

# for plotting
procs = []

# compare deadlines misses
def plot_func_dm(exps_dict):
    i=0
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    for exp_name, evals in exps_dict.items():
        for e in evals:
            x = [e['deadline_msec'] for e in evals]
            y = [e['deadlines_missed']/len(e['deadline_diffs'])*100. for e in evals]
        l2d = ax.plot(x, y, label=exp_name, 
            marker='.', markersize=10, markeredgewidth=0.7,
            c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        i+=1
        #ax.scatter(x, y, color=l2d[0].get_c())
    ax.invert_xaxis()
    ax.set_ylim(0., 105.)
    ax.legend(fontsize='medium')
    ax.set_ylabel('Deadline miss ratio (%)', fontsize='x-large')
    ax.set_xlabel('Deadline (msec)', fontsize='x-large')
    ax.grid('True', ls='--')
    #fig.suptitle("Ratio of missed deadlines over a range of deadlines", fontsize=16)
    plt.savefig("exp_plots/deadlines_missed.jpg")


procs.append(Process(target=plot_func_dm, args=(exps_dict,)))
procs[-1].start()


def plot_func_eted(exps_dict):
    i=0
    # compare execution times end to end
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    for exp_name, evals in exps_dict.items():
        x = [e['deadline_msec'] for e in evals]
        y = [e['exec_time_stats']['End-to-end'][1] for e in evals]
        l2d = ax.plot(x, y, label=exp_name,
            marker='.', markersize=10, markeredgewidth=0.7,
            c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        i+=1
        ax.scatter(x, y, color=l2d[0].get_c())
    ax.invert_xaxis()
    ax.set_ylim(.0, 140)
    ax.legend(fontsize='medium')
    ax.set_ylabel('End-to-end time (msec)', fontsize='x-large')
    ax.set_xlabel('Deadline (msec)', fontsize='x-large')
    ax.grid('True', ls='--')
    #fig.suptitle("Average end-to-end time over different deadlines", fontsize=16)
    plt.savefig("exp_plots/end-to-end_deadlines.jpg")


procs.append(Process(target=plot_func_eted, args=(exps_dict,)))
procs[-1].start()

# compare averaged AP of all classes seperately changing deadlines
def plot_avg_AP(merged_exps_dict):
    cls_per_file = 5
    cls_names = list(proto_AP_dict.keys())
    num_classes = len(cls_names)
    num_files = num_classes // cls_per_file + (num_classes % cls_per_file != 0)
    for filenum in range(num_files):
        if filenum == num_files-1:
            plot_num_classes = num_classes - (num_files-1) * cls_per_file
        else:
            plot_num_classes = cls_per_file
        fig, axs = plt.subplots(plot_num_classes, 1, \
                figsize=(12, 3*plot_num_classes), constrained_layout=True)
        cur_cls_names = cls_names[filenum*cls_per_file:filenum*cls_per_file+plot_num_classes]
        for ax, cls in zip(axs, cur_cls_names):
            for exp_name, evals in merged_exps_dict.items():
                x = evals['deadline_msec']
                y = evals['AP'][cls]['AP'] # for now, use AP as the only eval type as in nuscenes data
                y = [sum(e) / len(e) if len(e) > 0 else .0 for e in y ]
                l2d = ax.plot(x, y, label=exp_name)
                ax.scatter(x, y, color=l2d[0].get_c())
            ax.invert_xaxis()
            ax.legend(fontsize='medium')
            ax.set_ylabel(cls + ' AP', fontsize='large')
            ax.set_xlabel('Deadline (msec)', fontsize='large')
            ax.grid('True', ls='--')
        cur_cls_names_str = ""
        for s in cur_cls_names:
            cur_cls_names_str += s + ' '
        fig.suptitle(cur_cls_names_str + " classes, average precision over different deadlines", fontsize=16)
        plt.savefig(f"exp_plots/AP_deadlines_{filenum}.jpg")

#procs.append(Process(target=plot_avg_AP, \
#                     args=(merged_exps_dict,)))
#procs[-1].start()
def plot_stage_and_head_usage(merged_exps_dict):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
#    # axes[0] for num stages, axes[1] for heads, bar graph all, x axis deadlines
#    #calculate misses due to skipping heads
    i = 0
    for exp_name, evals in merged_exps_dict.items():
        x = evals['deadline_msec']
        y1, y2 = [], []
        for er in evals['rpn_stg_exec_seqs']:
            arr = [len(r[0]) for r in er]
            y1.append(sum(arr) / len(arr))
            arr = [len(r[1]) for r in er]
            y2.append(sum(arr) / len(arr))
        axes[0].plot(x, y1, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        axes[1].plot(x, y2, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
        i+=1

    ylim = 3.5
    for ax, ylbl in zip(axes, ('Avrg. RPN stages', 'Avrg. det heads')):
        ax.invert_xaxis()
        ax.legend(fontsize='medium')
        ax.set_ylabel(ylbl, fontsize='x-large')
        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
        ax.grid('True', ls='--')
        ax.set_ylim(0.0, ylim)
        ylim += 3.0

    plt.savefig("exp_plots/rpn_and_heads_stats.jpg")
#if plot_head_selection:
#    procs.append(Process(target=plot_stage_and_head_usage, \
#                         args=(merged_exps_dict,)))
#    procs[-1].start()

def plot_instance_data(merged_exps_dict):
    i=0
    # compare execution times end to end
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
    for ax, k in zip(axs, ['avrg_instances_detected', 'avrg_recognize_time']):
        for exp_name, evals in exps_dict.items():
            x = [e['deadline_msec'] for e in evals]
            y = [e[k] for e in evals]
            l2d = ax.plot(x, y, label=exp_name,
                marker='.', markersize=10, markeredgewidth=0.7,
                c=evals[0]['color'], linestyle=evals[0]['lnstyle'])
            i+=1
            ax.scatter(x, y, color=l2d[0].get_c())
        ax.invert_xaxis()
        ax.set_ylim(.0, 140)
        ax.legend(fontsize='medium')
        ax.set_ylabel(k, fontsize='x-large')
        ax.set_xlabel('Deadline (msec)', fontsize='x-large')
        ax.grid('True', ls='--')
    #fig.suptitle("Average end-to-end time over different deadlines", fontsize=16)
    plt.savefig("exp_plots/instance_data.jpg")

for exp_name, evals in merged_exps_dict.items():
    evals['mAP']['normalized_NDS'] = np.array(evals['mAP']['NDS']) / max_NDS * 100.
   
#
#procs.append(Process(target=plot_instance_data, \
#                     args=(merged_exps_dict,)))
#procs[-1].start()
#
#Add normalized accuracy
i = 0
fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
for exp_name, evals in merged_exps_dict.items():
    x = evals['deadline_msec']
    y = evals['mAP']['normalized_NDS']
    l2d = ax.plot(x, y, label=exp_name,
            marker='.', markersize=10, markeredgewidth=0.7,
            c=evals['color'][0], linestyle=evals['lnstyle'][0])
    i+=1
ax.invert_xaxis()
ax.legend(fontsize='medium')
ax.set_ylabel('Normalized accuracy (%)', fontsize='x-large')
ax.set_xlabel('Deadline (msec)', fontsize='x-large')
ax.grid('True', ls='--')
ax.set_ylim(0.0, 105.)

plt.savefig("exp_plots/normalized_NDS_deadlines.jpg")

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
labels = list(merged_exps_dict.keys())
x_values = np.arange(len(labels))
y_values = [round(sum(evals['mAP']['normalized_NDS'])/ \
        len(evals['mAP']['normalized_NDS']),1) \
        for evals in merged_exps_dict.values()]

rects = ax.bar(x_values, y_values, color=[v['color'][0] for v in merged_exps_dict.values()])
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

autolabel(rects)
for r, l in zip(rects, labels):
    r.set_label(l)
ax.legend(fontsize='medium', ncol=2)
ax.set_ylabel('Average accuracy (%)', fontsize='x-large')
#ax.set_xlabel(')', fontsize='x-large')
#ax.grid('True', ls='--')
ax.set_ylim(0.0, 105.)

plt.savefig("exp_plots/normalized_NDS_bar.jpg")

for p in procs:
    p.join()

sys.exit(0)
#####################################################################################
## compare mAP for all types
#i = 0
#fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
#for ax, eval_type in zip(axs, proto_mAP_dict.keys()):
#    for exp_name, evals in merged_exps_dict.items():
#        x = evals['deadline_msec']
#        y = evals['mAP'][eval_type]
#        l2d = ax.plot(x, y, label=exp_name, linestyle=linestyles[i])
#        ax.scatter(x, y, color=l2d[0].get_c())
#        i+=1
#    ax.invert_xaxis()
#    ax.legend(fontsize='small', ncol=2)
#    ax.set_ylabel(eval_type, fontsize='large')
#    ax.set_xlabel('Deadline (msec)', fontsize='large')
#    ax.grid('True', ls='--')
#    ax.set_ylim(0.0, 1.0)
#
#fig.suptitle("Accuracy over different deadlines", fontsize=16)
#plt.savefig("exp_plots/mAP_deadlines.jpg")
#


#def plot_impr_rem_hist(exps_dict):
#    for exp_name, exp_dict in exps_dict.items():
#        if 'imprecise' not in exp_name:
#            continue
#        fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
#        for i, ax in enumerate(axs):
#            dist = exp_dict['exec_times'][f"impr{i + 1}"]
#            perc99 = np.percentile(dist, 99, interpolation='lower')
#            perc1 = np.percentile(dist, 1, interpolation='lower')
#            dist_ = [et for et in dist if et < perc99 and et > perc1]
#            ax.hist(dist_, 33)
#            ax.set_xlabel(f"Remaining time to execute {i} stages (ms, 99pct)")
#            ax.text((perc99 + perc1) / 2, 20, f"Max: {perc99:.5}")
#            print(f"Remaining time to execute {i + 1} stages is 99 perc:{perc99}")
#            print(f"Remaining time to execute {i + 1} stages is max:", max(dist))
#
#        fig.suptitle("Remaining time historgram for three imprecise stage cases", fontsize=16)
#        plt.savefig("exp_plots/impr_hist.jpg")
#
#
#procs.append(Process(target=plot_impr_rem_hist, \
#                     args=(exps_dict,)))
#procs[-1].start()
#
##
#for exp_name, exp_dict in exps_dict.items():
#    if diff_slc:
#        # RPN scaling
#        fig, axs = plt.subplots(4, 1, figsize=(16, 12))
#        x = exp_dict['slice_size_perc']
#        # x = exp_dict['stage1_slice_size']
#        for stg, ax in enumerate(axs[:-2]):
#            stg_key = f"RPN-stage-{stg + 2}"
#            mam_dict = exp_dict['exec_times']
#            # last one does not do slicing
#            y = [mam_dict[stg_key][1][i] / mam_dict[stg_key][1][-1] * 100 - x[i] \
#                 for i in range(len(mam_dict[stg_key][1]) - 1)]
#            ax.bar(x[:-1], y, width=0.4)
#            ax.set_ylabel(f'RPN stage {stg + 2} time scale error percentage')
#            ax.set_xticks(list(range(10, 65, 5)))
#            ax.set_xlabel('Slice size percentage')
#
#        axs[-2].scatter(x, exp_dict['stage1_slice_size'])
#        axs[-2].set_ylabel('Stage 1 slice height')
#        axs[-2].set_xlabel('Slice size percentage')
#        axs[-2].set_xticks(list(range(10, 105, 5)))
#        axs[-2].grid('True', ls='--')
#
#        axs[-1].scatter(x, exp_dict['num_slices'])
#        axs[-1].set_ylabel('Number of slices')
#        axs[-1].set_xlabel('Slice size percentage')
#        axs[-1].set_xticks(list(range(10, 105, 5)))
#        axs[-1].grid('True', ls='--')
#
#        fig.suptitle("RPN execution time scaling", fontsize=16)
#        plt.savefig("exp_plots/rpn_scaling_perc.jpg")
#
#        # num slices vs slice size
# better approach: use plot dict
#def fill_plot_dict(plot_dict, exp_dict, keys, pred_keys=None):
#    if pred_keys is None:
#        pred_keys = list()
#    for exp in exp_dict.keys():
#        if exp not in plot_dict:
#            plot_dict[exp] = {}
#        for k in keys:
#            if len(pred_keys) == 0:
#                plot_dict[exp][k] = exp_dict[exp][k]
#            elif len(pred_keys) == 1:
#                plot_dict[exp][k] = \
#                    exp_dict[exp][pred_keys[0]][k]
#            elif len(pred_keys) == 2:
#                plot_dict[exp][k] = \
#                    exp_dict[exp][pred_keys[0]][pred_keys[1]][k]
#
#plot_dict = {}
#exec_keys_to_plot = [
#    'PreProcess', 'RPN-stage-1', 'RPN-stage-2', 'RPN-stage-3',
#    'RPN-finalize', 'PostProcess', 'End-to-end']
#
#fill_plot_dict(plot_dict, merged_exps_dict, exec_keys_to_plot, ['exec_times'])
#procs.append(Process(target=plot_func_hist, args=(plot_dict,  "ALL_", )))
#procs[-1].start()
#plot_dict.clear()

#exec_keys_to_plot = ['PFE', 'PillarGen', 'PillarPrep',
#                     'PillarFeatureNet', 'PillarScatter']
#fill_plot_dict(plot_dict, merged_exps_dict, exec_keys_to_plot, ['exec_times'])
#fill_plot_dict(plot_dict, merged_exps_dict, ['num_voxels'],)
#procs.append(Process(target=plot_func_hist, args=(plot_dict,  "PFE_", )))
#procs[-1].start()
#
#exec_keys_to_plot = ['exec_keys_to_plot']

#def plot_func_sorted(plot_dict, x_key, filename_prefix):
#    for exp_name, merged_evals in plot_dict.items():
#        h = (len(merged_evals) - 1) // 2
#        fig, axs = plt.subplots(h, 2, figsize=(16, h * 4))
#        axs = np.concatenate(axs)
#
#        x = np.concatenate(merged_evals[x_key])
#        del merged_evals[x_key]
#        sorted_indexes = np.argsort(x)
#        # downsample indexes until array length is small enough
#        target_size = 250
#        if len(sorted_indexes) > target_size:
#            mask = [True] + ([False] * math.floor(len(sorted_indexes) // target_size))
#            mask = mask * math.ceil(len(sorted_indexes) / len(mask))
#            sorted_indexes = sorted_indexes[mask[:len(sorted_indexes)]]
#
#        x = x[sorted_indexes]
#
#        for (i, ax), (key, val) in zip(enumerate(axs), merged_evals.items()):
#            # Use 99 percentile
#            y = np.concatenate(val)
#            if len(y) > 0:
#                y = y[sorted_indexes]
#                perc99_idx = int(len(y) * 0.99)
#                print(f"{exp_name} {key} 99 perc:{y[perc99_idx]}, max: {max(x)}")
#                ax.plot(x[:perc99_idx], y[:perc99_idx])
#                ax.grid('True', ls='--')
#                ax.set_ylabel(key + ' execution time (ms)')
#            ax.set_xlabel(x_key)
#            # #plot first instance timeline
#            # ax.bar(np.arange(len(mrg_exe_times[key][0])), mrg_exe_times[key][0])
#            # ax.set_xlabel('deadline ' + str(merged_evals['deadline_msec'][0]) + key +
#            #               ' execution timeline (sample ID)')
#
#        fig.suptitle(exp_name + " - Number of voxels vs. Execution Time of PFE Phases", fontsize=16)
#        plt.savefig("exp_plots/" + filename_prefix + exp_name + "_voxel_pfe_bar.jpg")
#
## plot dict was already filled except end to end
#fill_plot_dict(plot_dict, merged_exps_dict, ['End-to-end'], ['exec_times'])
#procs.append(Process(target=plot_func_sorted, args=(plot_dict, 'num_voxels', "", )))
#procs[-1].start()


#def plot_gt_miss(imp_merged_evals, method):
#    if 'gt_counts' not in imp_merged_evals:
#        print('Skipping head/miss analysis')
#        return
#
#    #calculate misses due to skipping heads
#    misses = []
#    for exp_gt_counts, exp_seqs in zip(imp_merged_evals['gt_counts'], \
#            imp_merged_evals['rpn_stg_exec_seqs']):
#        misses.append([0] * len(exp_gt_counts[0]))
#        for gt_count, seq in zip(exp_gt_counts, exp_seqs):
#            for s in seq[1]:
#                gt_count[s] = 0
#            for i, g in enumerate(gt_count):
#                misses[-1][i] += 1 if g else 0
#    misses = np.transpose(misses)
#    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
#
#    x = imp_merged_evals['deadline_msec']
#    for i, y in enumerate(misses):
#        l2d = ax.plot(x, y, label=f'head {i+1}')
#        ax.scatter(x, y, color=l2d[0].get_c())
#        ax.invert_xaxis()
#    ax.legend(fontsize='medium')
#    ax.set_ylabel('Misses', fontsize='large')
#    ax.set_xlabel('Deadline (msec)', fontsize='large')
#    ax.grid('True', ls='--')
#
#    fig.suptitle("Objects missed due to skipping heads", fontsize=16)
#    plt.savefig(f"exp_plots/gtmiss_deadlines_{method}.jpg")
#
#procs.append(Process(target=plot_gt_miss, \
#                     args=(merged_exps_dict['Imprecise'],'Imprecise',)))
#procs[-1].start()
#procs.append(Process(target=plot_gt_miss, \
#                     args=(merged_exps_dict['Imprecise-NANP'],'Imprecise-NANP',)))
#procs[-1].start()

#
#def plot_func_hist(plot_dict, filename_prefix):
#    for exp_name, merged_evals in plot_dict.items():
#        h, w = len(merged_evals) // 2, 2
#        if h == 0:
#            h, w = 1, 1
#        fig, axs = plt.subplots(h, w, figsize=(16, h * 4))
#        if w == 1:
#            axs = [axs]
#        axs = np.concatenate(axs)
#
#        for (i, ax), (key, val) in zip(enumerate(axs), merged_evals.items()):
#            # Use 99 percentile
#            x = np.concatenate(val)
#            if len(x) > 0:
#                perc99 = np.percentile(x, 99, interpolation='lower')
#                # print(f"{exp_name} {key} 99 perc:{perc99}, max: {max(x)}")
#                x = [et for et in x if et < perc99]
#                ax.hist(x, 33)
#            ax.set_xlabel(key + ' execution time bins (ms)')
#            # #plot first instance timeline
#            # ax.bar(np.arange(len(mrg_exe_times[key][0])), mrg_exe_times[key][0])
#            # ax.set_xlabel('deadline ' + str(merged_evals['deadline_msec'][0]) + key +
#            #               ' execution timeline (sample ID)')
#
#        fig.suptitle("Execution time histogram of " + exp_name, fontsize=16)
#        plt.savefig("exp_plots/" + filename_prefix + exp_name + "_exec_time_hist.jpg")


