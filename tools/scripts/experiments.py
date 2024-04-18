import os
import glob
import subprocess
import time

# Initialize paths and names
experiment_name = 'iterative_gnn_1'

BASE_PATH = '/root/OpenPCDet/tools'
base_path = "./cfgs/kitti_models"
experiments_input_path = os.path.join(base_path, "experiments", experiment_name)

# Initialize output paths
kitti_output_path = "../output/cfgs/kitti_models"
experiment_output_path = os.path.join(kitti_output_path, "experiments", experiment_name)

# Read all model config paths from the directory
yaml_files = glob.glob(os.path.join(experiments_input_path, "*.yaml"))
experiments = [os.path.splitext(os.path.basename(f))[0] for f in yaml_files]
# sort the experiments 
# experiments = sorted(experiments, key=lambda x: int(x.split('_')[0]))

# print the experiments
print(f"Running experiments at {experiments_input_path}: {experiments}")

# create tensorboard directory
os.makedirs(os.path.join(experiment_output_path, 'tensorboard'), exist_ok=True)
cmd = f'(cd {experiment_output_path}; tensorboard dev upload --logdir tensorboard --name "{experiment_name}")'
tensorboard_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      
# Run all experiments
for e in experiments:
    success = False
    while not success:
        try:
            config_path = os.path.join(experiments_input_path, f"{e}.yaml")
            cmd = ["python", "train.py", "--cfg", config_path, "--extra_tag", f"{e}"]
            subprocess.run(cmd, check=True, cwd=BASE_PATH)
            
            source_path = os.path.join(experiment_output_path, f"{e}/{e}/tensorboard")
            tensor_file = glob.glob(os.path.join(source_path, "*"))
            dest_path = os.path.join(experiment_output_path, 'tensorboard', f"{e}")
            os.makedirs(dest_path, exist_ok=True)

            # Get the base name of the source file
            file_name = os.path.basename(tensor_file[0])

            # Create the full destination path including the file name
            full_dest_path = os.path.join(dest_path, file_name)

            # Perform the copy
            os.system(f"cp -r {tensor_file[0]} {full_dest_path}")
            print(f"Finished experiment {e}")
            success = True
        except Exception as ex:
            print(f"Error while running experiment {e}: {ex}")
            time.sleep(60)
            print(f"Retrying experiment {e}")

# Kill tensorboard process
# wait for 10 minutes to make sure tensorboard is done
time.sleep(600)
tensorboard_process.terminate()
