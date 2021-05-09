from kubernetes import client
from kfserving import KFServingClient
from kfserving import constants
from kfserving import V1beta1InferenceService
from kfserving import V1beta1InferenceServiceSpec
from kfserving import V1beta1PredictorSpec
from kfserving import V1beta1TFServingSpec

from kubernetes.client import V1ResourceRequirements

from constants import MODEL_NAME

storage_uri = 's3://servedmodels/ssd_inception_v2_coco/'
namespace = 'kubeflow-user'
service_account_name = 'sa'

isvc = V1beta1InferenceService(api_version=f'{constants.KFSERVING_GROUP}/{constants.KFSERVING_V1BETA1_VERSION}',
                               kind=constants.KFSERVING_KIND,
                               metadata=client.V1ObjectMeta(
                                   name=MODEL_NAME, namespace=namespace),
                               spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                       min_replicas=1,
                                       service_account_name=service_account_name,
                                       tensorflow=(V1beta1TFServingSpec(
                                           resources=V1ResourceRequirements(
                                               requests={'cpu': '100m', 'memory': '1Gi'},
                                               limits={'cpu': '100m', 'memory': '1Gi'}),
                                           runtime_version='1.15.0',
                                           storage_uri=storage_uri)))),
                               )

KFServing = KFServingClient()

if True:
    KFServing.create(isvc)
    KFServing.get(MODEL_NAME, namespace=namespace, watch=True, timeout_seconds=300)
else:
    print(KFServing.delete(MODEL_NAME, namespace=namespace))
