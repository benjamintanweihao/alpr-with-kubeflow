import os

import kfp
from kfp import dsl, components
from kfp.components import InputPath
from kfp.dsl import PipelineVolume
from kubernetes.client import V1EnvVar

PROJECT_ROOT = '/workspace/alpr-with-kubeflow'


def add_env_variables(op):
    op.container.add_env_variable(V1EnvVar(name='TF_FORCE_GPU_ALLOW_GROWTH', value="true"))
    return op


# A couple of things needs to happen before this would work
# 1. Create a namepace with `kfserving-inference-service` and
#    with the `serving.kubeflow.org/inferenceservice=enabled` label.
#    This namespace shouldn't have a `control-plane` level
# 2. Within this namespace, you would need to create two things:
# a) A Secret that would contain the MinIO credentials
# b) A ServiceAccount that points to this Secret.
#
# The Serving Op will then reference this ServiceAccount in order to
# access the MinIO credentials.

def serving_op(export_bucket: str, model_name: str, model_dns_prefix: str):
    kfserving_op = components.load_component_from_url(
        'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/kfserving/component.yaml'
    )

    op = kfserving_op(
        action="create",
        default_model_uri=f"s3://{export_bucket}/{model_name}",
        model_name=model_dns_prefix,  # this should be DNS friendly
        namespace='kfserving-inference-service',
        framework="tensorflow",
        service_account="sa"
    )

    return op


@dsl.pipeline(
    name='Serving Pipeline',
    description='This is a single component Pipeline for Serving'
)
def alpr_pipeline(
        model_name: str = 'saved_model_half_plus_two_cpu',
        export_bucket: str = 'servedmodels',
        model_dns_prefix: str = 'ssd-inception-v2'):
    serving_op(export_bucket=export_bucket,
               model_name=model_name,
               model_dns_prefix=model_dns_prefix)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'serving-pipeline.zip')
