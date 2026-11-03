# Training and Testing

## NuScenes 3D Semantic Occupancy Prediction Task
**a. Train TFusionOcc with 8 GPUs.**
```shell
bash tools/dist_train.sh configs/${path to any config file}$ 8
```
**b. Test TFusionOcc with 8 GPUs.**
```shell
bash tools/dist_test.sh configs/${path to any config file}$ ${path to corresponding checkpoint}$ 8
```
