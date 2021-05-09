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

Assume that you have a MinIO bucket called `servedmodels` with the following files:

```bash
servedmodels → tree
└── ssd_inception_v2_coco
    └── 1
        ├── checkpoint
        ├── frozen_inference_graph.pb
        ├── model.ckpt.data-00000-of-00001
        ├── model.ckpt.index
        ├── model.ckpt.meta
        ├── saved_model.pb
        └── ssd_inception_v2_coco.config
```

Step 1: Apply the MinIO secrets:

```bash
% kubectl apply -f serving/minio_s3_secrets.yaml
```

Step 2: Apply the Auth Policy patch:

```bash
% kubectl apply -f serving/kubeflow_user_auth_policy.yaml
```

Step 3: Find the IP address of the pod:

```bash
% kubectl get  po -l serving.kubeflow.org/inferenceservice=ssd-inception-v2 -n kubeflow-user -o wide
NAME                                                              READY   STATUS    RESTARTS   AGE    IP           NODE      NOMINATED NODE   READINESS GATES
ssd-inception-v2-predictor-default-00001-deployment-f948576bfjj   3/3     Running   0          174m   10.42.0.81   artemis   <none>           <none>
```

Step 4: The service would be available on http://10.42.0.81:8080/v1/models/ssd-inception-v2

For example:

```buildoutcfg
% curl http://localhost:8080/v1/models/ssd-inception-v2
```

This should return:

```json
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```