
## Installation
- This Docker Image is based on cuda 11.6 and torch 1.13.1. If your nvidia-drive is able to support this version, just follow the below guidance

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet

# For Making Docker Image
./scripts/make_image.sh

# For Making Docker Image without cache
./scripts/make_image.sh --nocache

```

## Usage

- You can launch docker image after finishing the Installation

```bash
# For Launching Docker Image temporarily
./scripts/run_image.sh

# For MakLaunchinging Docker Image permanently
./scripts/run_image.sh --keep-alive
```
