from shared import pcdet_cuda_image, volumes
import modal

app = modal.App("pcdet")

@app.function(image=pcdet_cuda_image, gpu="T4", timeout=3600*2, volumes=volumes)
def build_nuscenes():
    print("-> Building nuScenes")

    import subprocess
    subprocess.run(["python", "-m", "pcdet.datasets.nuscenes.nuscenes_dataset", "--func", "create_nuscenes_infos", 
    "--cfg_file", "tools/cfgs/dataset_configs/nuscenes_dataset.yaml", "--version", "v1.0-mini"])

    print("-> Done building nuScenes")

@app.local_entrypoint()
def main():
    build_nuscenes.remote()