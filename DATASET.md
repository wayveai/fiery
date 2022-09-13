## 📖 Dataset

- Go to the official website of the [NuScenes dataset](https://www.nuscenes.org/download). You will need to create an 
account. 
- Download the _Full dataset (v1.0)_. This includes the _Mini_ dataset and the _Trainval_ dataset with all the 
metadata and sensor file blobs.
- Extract the tar files in order to obtain the following folder structure. The `nuscenes` dataset folder will be 
designated as `${NUSCENES_DATAROOT}`.
```
nuscenes  
│
└───trainval
│     maps
│     samples
│     sweeps
│     v1.0-trainval
│   
└───mini
      maps
      samples
      sweeps
      v1.0-mini
```
