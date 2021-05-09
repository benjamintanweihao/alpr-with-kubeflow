# Model Serving

## To test on tensorflow/serving:

```
% docker run -t --rm -p 8500:8500 -p 8501:8501 -v \
 "/home/benjamintan/workspace/servedmodels/ssd_inception_v2_coco_2018_01_28/:/models/ssdv2" -e \
 MODEL_NAME=ssd-inception-v2  tensorflow/serving:1.15.0
```

 HOST = "localhost"
 PORT = "8501"

## To test on KFServing:

Step 1: Port-forward on the KFServing pod:

```bash
% kubectl port-forward ssd-inception-v2-predictor-default-00001-deployment-77d4cdlvncv --namespace kubeflow-user 8080:8012
```

Step 2: The service would be available on http://localhost:8080/v1/models/ssd-inception-v2

For example:

```buildoutcfg
% curl http://localhost:8080/v1/models/ssd-inception-v2/metadata
```
