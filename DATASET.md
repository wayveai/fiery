## 📖 Dataset

- Go to the official website of the [NuScenes dataset](https://www.nuscenes.org/download). You will need to create an 
account. 
- Download the _Full dataset (v1.0)_. This includes the _Mini_ dataset (Metadata and sensor file blobs) and the 
_Trainval_ dataset (Metadata, File blobs part 1-10).
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
- The full dataset is around ~400GB. It is possible to reduce the dataset size to ~60GB by only downloading the 
keyframe blobs only part 1-10, instead of all the file blobs part 1-10. The keyframe blobs contain the data we 
need (RGB images and 3D bounding box of dynamic objects at 2Hz). The remaining file blobs also include RGB 
images and LiDAR sweeps at a higher frequency, but they are not used during training. 