import modal

pcdet_cuda_image = (
    modal.Image.from_dockerfile("current.dockerfile")
    .copy_local_dir("../tools", "/OpenPCDet/tools")
)

volume = modal.Volume.from_name("nuscenes")
VOL_MOUNT_PATH = "/OpenPCDet/data/nuscenes"

volumes={VOL_MOUNT_PATH: volume}