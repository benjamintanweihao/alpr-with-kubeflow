import argparse
from kubernetes import client
from kfserving import KFServingClient
from kfserving import constants
from kfserving import V1beta1InferenceService
from kfserving import V1beta1InferenceServiceSpec
from kfserving import V1beta1PredictorSpec
from kfserving import V1beta1TFServingSpec

from kubernetes.client import V1ResourceRequirements

from constants import MODEL_NAME


def create_inference_service(namespace: str,
                             name: str,
                             storage_uri: str,
                             runtime_version: str,
                             service_account_name: str):
    isvc = V1beta1InferenceService(api_version=f'{constants.KFSERVING_GROUP}/{constants.KFSERVING_V1BETA1_VERSION}',
                                   kind=constants.KFSERVING_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=name, namespace=namespace),
                                   spec=V1beta1InferenceServiceSpec(
                                       predictor=V1beta1PredictorSpec(
                                           min_replicas=1,
                                           service_account_name=service_account_name,
                                           tensorflow=(V1beta1TFServingSpec(
                                               resources=V1ResourceRequirements(
                                                   requests={'cpu': '100m', 'memory': '1Gi'},
                                                   limits={'cpu': '100m', 'memory': '1Gi'}),
                                               runtime_version=runtime_version,
                                               storage_uri=storage_uri)))),
                                   )

    KFServing = KFServingClient()

    KFServing.create(isvc)
    KFServing.get(MODEL_NAME, namespace=namespace, watch=True, timeout_seconds=300)
    # print(KFServing.delete(MODEL_NAME, namespace=namespace))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--namespace",
                        help="namespace to deploy the inference service to",
                        type=str,
                        default='kubeflow-user-example-com')

    parser.add_argument("--name",
                        help="name of inference service",
                        default='ssd-inception-v2',
                        type=str)

    parser.add_argument("--storage_uri",
                        help="storage_uri of model. e.g. s3://servedmodels/ssd_inception_v2_coco/SAVED_MODEL/ssd_inception_v2_coco",
                        default="s3://servedmodels/ssd_inception_v2_coco/SAVED_MODEL/ssd_inception_v2_coco",
                        type=str)

    parser.add_argument("--runtime_version",
                        help="version of Tensorflow. e.g. 1.15.0 or 1.15.0-gpu",
                        type=str,
                        default='1.15.0')

    parser.add_argument("--service_account_name",
                        help="Service Account name that stores the credentials",
                        type=str,
                        default='sa')

    args = parser.parse_args()

    create_inference_service(namespace=args.namespace,
                             name=args.name,
                             storage_uri=args.storage_uri,
                             runtime_version=args.runtime_version,
                             service_account_name=args.service_account_name)
