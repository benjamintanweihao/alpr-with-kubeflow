import os

from kfp import dsl
from kfp.dsl import PipelineVolume
from kubernetes.client import V1EnvVar

PROJECT_ROOT = '/workspace'


def add_env_variables(op):
    op.container.add_env_variable(V1EnvVar(name='PYTHONPATH', value=f"{os.path.join(PROJECT_ROOT, 'alpr')}"))
    op.container.add_env_variable(V1EnvVar(name='TF_FORCE_GPU_ALLOW_GROWTH', value="true"))
    return op


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
        "mkdir ~/.ssh",
        "cp /etc/ssh-key/id_rsa ~/.ssh/id_rsa",
        "chmod 600 ~/.ssh/id_rsa",
        f"git clone git@github.com:benjamintanweihao/alpr-with-kubeflow.git {PROJECT_ROOT}",
        f"cd {PROJECT_ROOT}/alpr-with-kubeflow",
        f"git checkout {branch_or_sha}",
    ]

    op = dsl.ContainerOp(
        name='git clone project',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


def convert_to_tfrecords_op(image, pvolume):
    pass


def train_and_eval_op(image, pvolume):
    pass


@dsl.pipeline(
    name='ALPR Pipeline',
    description='This is the ALPR Pipeline that is meant to be executed on KubeFlow.'
)
def alpr_pipeline(
        image: str = "benjamintanweihao/alpr-kubeflow",
        branch: str = 'master'):

    resource_name = 'alpr-pipeline-pvc'
    # 0. create pipeline volume
    create_pipeline_volume = create_pipeline_volume_op(resource_name=resource_name)

    git_clone = git_clone_op(branch_or_sha=branch,
                             pvolume=create_pipeline_volume.volume)

    # DATA PREP
    convert_to_tfrecords = convert_to_tfrecords_op(
        image=image,
        pvolume=git_clone.pvolume,
    ).after(git_clone)

    _training_and_eval = train_and_eval_op(image=image,
                                           pvolume=convert_to_tfrecords.pvolume)
