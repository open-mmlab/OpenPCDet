
# Anytime-Lidar

Anytime-lidar deliver anytime perception to lidar-based object detection DNNs. It is implemented on top of the OpenPCDet project (https://github.com/open-mmlab/OpenPCDet). The entire testing environment is packaged in a docker image, which needs to be pulled to Jetson AGX Xavier. The Jetpack version we used is as follows. Make sure yours is not less than this.
```
nvidia@devboard-xavier:~$ cat /etc/nv_tegra_release 
# R32 (release), REVISION: 6.1, GCID: 27863751, BOARD: t186ref, EABI: aarch64, DATE: Mon Jul 26 19:36:31 UTC 2021
```
Another thing is that we maximized all CPU and GPU clocks for testing, so please do that as well using `jetson_clocks` and `nvpmodel` tools available by NVIDIA.

Pull the docker image:
```
docker pull kucsl/pointpillars:1.1.0
```
NOTE: The docker image has a size of 20.5 GiB. It has the mini nuScenes dataset included. To pull it to the Jetson AGX Xavier, an external storage has to be attached to the module (unless you have enough space), and the docker should be configured to store the images on the external storage, which is not default. Please follow the steps in the following link on how to do this: https://evodify.com/change-docker-storage-location/ (Change Docker storage location: THE RIGHT WAY!)

After pulling the image, The following command can be used to start a container and attach to it.
```
docker run --runtime nvidia -it --privileged --cap-add=ALL  --ulimit rtprio=99 --tmpfs /tmpfs --name pp kucsl/pointpillars:1.1.0
```
Then run the following command one by one to make calibration and run the tests. When the last command is done, the results will be plotted and saved at the location `~/OpenPCDet/tools/exp_plots`. You can copy the plots to your local computer to check them.
```
cd ~/OpenPCDet/tools
. env.sh
. nusc_dataset_prep.sh
. calib_and_run_tests.sh
```


## Citation 
If you find this project useful in your research, please consider cite:

```
inproceedings{anytimelidar2022,
  author    = {Ahmet Soyyigit and
               Shuochao Yao and
               Heechul Yun},
  title     = {Anytime-Lidar: Deadline-aware 3D Object Detection},
  booktitle = {28th {IEEE} International Conference on Embedded and Real-Time Computing
               Systems and Applications, {RTCSA} 2028, Taipei, Taiwan, August 23-25,
               2022},
  publisher = {{IEEE}},
  year      = {2022},
}
```

