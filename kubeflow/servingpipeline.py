import kfp
from kfp import dsl
from kubernetes.client import V1EnvVar

PROJECT_ROOT = '/workspace/alpr-with-kubeflow'


def add_env_variables(op):
    op.container.add_env_variable(V1EnvVar(name='AWS_ACCESS_KEY_ID', value="minio"))
    op.container.add_env_variable(V1EnvVar(name='AWS_SECRET_ACCESS_KEY', value="minio123"))
    op.container.add_env_variable(V1EnvVar(name='S3_USE_HTTPS', value="false"))

    return op


# A couple of things needs to happen before this would work
# 1. Create a namepace with `kfserving-inference-service` and
#    with the `serving.kubeflow.org/inferenceservice=enabled` label.
#    This namespace shouldn't have a `control-plane` level
#
# 2. Within this namespace, you would need to create two things:
# a) A Secret that would contain the MinIO credentials
# b) A ServiceAccount that points to this Secret.
#
# 3. Modify the InferenceService ConfigMap to include the
#    tensorflow-1.15 and tensorflow-gpu-1.15.
#
# The Serving Op will then reference this ServiceAccount in order to
# access the MinIO credentials.

def serving_op(export_bucket: str, model_name: str, model_dns_prefix: str, model_version: str):
    kfserving_op = kfp.components.load_component_from_url(
        'https://raw.githubusercontent.com/kubeflow/pipelines/1.5.0/components/kubeflow/kfserving/component.yaml')

    op = kfserving_op(
        model_name=model_dns_prefix,  # this should be DNS friendly
        model_uri=f"s3://{export_bucket}/{model_name}/1/{model_version}",
        namespace='kubeflow',
        framework="tensorflow",
        service_account="sa",
        min_replicas='1',
        max_replicas='1',
        watch_timeout='360',
        autoscaling_target='1',
    )

    op = add_env_variables(op)

    return op


@dsl.pipeline(
    name='Serving Pipeline',
    description='This is a single component Pipeline for Serving'
)
def alpr_pipeline(
        model_name: str = 'ssd_inception_v2_coco',
        export_bucket: str = 'servedmodels',
        model_dns_prefix: str = 'ssd-inception-v2',
        model_version: str = '1',
):
    _ = serving_op(
        export_bucket=export_bucket,
        model_name=model_name,
        model_dns_prefix=model_dns_prefix,
        model_version=model_version)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'serving-pipeline.zip')
