import kfp
from kfp import dsl
from kfp.dsl import PipelineVolume

PROJECT_ROOT = '/workspace/alpr-with-kubeflow'


def create_pipeline_volume_op(resource_name):
    return dsl.VolumeOp(
        name="create-pipeline-volume",
        resource_name=resource_name,
        modes=dsl.VOLUME_MODE_RWO,
        size="10Gi",
    )


def git_clone_op(branch_or_sha: str, pvolume: PipelineVolume):
    image = 'benjamintanweihao/alpr-vc'

    commands = [
        f"git clone https://github.com/benjamintanweihao/alpr-with-kubeflow.git {PROJECT_ROOT}",
        f"cd {PROJECT_ROOT}",
        f"git checkout {branch_or_sha}",
    ]

    for c in commands:
        print(c)

    op = dsl.ContainerOp(
        name='git clone project',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


def serving_op(image: str, pvolume: PipelineVolume) -> object:
    commands = [
        f'cd {PROJECT_ROOT}',
        f'python serving/kfs_deployer.py'
    ]

    for c in commands:
        print(c)

    op = dsl.ContainerOp(
        name='serve model',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


@dsl.pipeline(
    name='Serving Pipeline',
    description='This is a single component Pipeline for Serving'
)
def alpr_pipeline(
        image: str = "benjamintanweihao/alpr-kubeflow",
        branch: str = 'master'):

    resource_name = 'alpr-pipeline-pvc'

    create_pipeline_volume = create_pipeline_volume_op(resource_name=resource_name)

    git_clone = git_clone_op(branch_or_sha=branch,
                             pvolume=create_pipeline_volume.volume)

    _ = serving_op(image=image, pvolume=git_clone.pvolume)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'serving-pipeline.zip')
