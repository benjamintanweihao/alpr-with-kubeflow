import kfp
from kfp import dsl

from kubeflow.components import create_pipeline_volume_op, git_clone_op, serving_op


@dsl.pipeline(
    name='Serving Pipeline',
    description='This is a single component Pipeline for Serving'
)
def alpr_pipeline(
        image: str = "benjamintanweihao/alpr-kubeflow",
        branch: str = 'master',
        model_dns_prefix: str = 'ssd-inception-v2',
):
    resource_name = 'alpr-pipeline-pvc'

    create_pipeline_volume = create_pipeline_volume_op(resource_name=resource_name)

    git_clone = git_clone_op(branch_or_sha=branch,
                             pvolume=create_pipeline_volume.volume)

    _ = serving_op(image=image,
                   model_dns_prefix=model_dns_prefix,
                   pvolume=git_clone.pvolume)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'serving-pipeline.zip')
