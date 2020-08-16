import os
import time

import kfp
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
        f"git clone https://github.com/benjamintanweihao/alpr-with-kubeflow.git {PROJECT_ROOT}",
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


def convert_to_tfrecords_op(image: str, pvolume: PipelineVolume):
    commands = []
    datasets = ['indian', 'romanian', 'voc']
    for d in datasets:
        for split in ['train', 'test']:
            input_dir = os.path.join('DATASETS', d, split, 'annotations')
            commands.append(
                f'python dataprep/scripts/xml2csv.py -i {input_dir} -o {os.path.join(input_dir, "train_labels.csv")}',
            )

    for d in datasets:
        for split in ['train', 'test']:
            csv_input = os.path.join('DATASETS', d, split, 'annotations', f'{split}_labels.csv')
            img_path = os.path.join('DATASETS', d, split, 'images')
            commands.append(
                f'python dataprep/scripts/csv2tfrecords.py --split_name={split} --tfrecord_name={d} '
                f'--label=license-plate --csv_input={csv_input} --output_path TFRECORDS/ --img_path={img_path}'
            )

    op = dsl.ContainerOp(
        name='convert to tfrecords',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


def train_and_eval_op(image: str, pvolume: PipelineVolume, model_name: str):
    model_dir = f'LOGS/{model_name}{int(time.time())}'

    commands = [
        f'python model_main.py --model_dir {model_dir} --pipeline_config_path ../model_configs/{model_name}.config'
    ]

    op = dsl.ContainerOp(
        name='convert to tfrecords',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


def export_save_model_op(image: str, pvolume: PipelineVolume, model_name: str):
    commands = [
        f'python MODELS/research/object_detection/export_inference_graph.py --input_type image_tensor '
        f'--pipeline_config_path train/model_configs/{model_name}.config '
        f'--trained_checkpoint_prefix LOGS/model.ckpt --output_directory SAVED_MODEL/{model_name}'
    ]

    op = dsl.ContainerOp(
        name='export savedmodel',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


@dsl.pipeline(
    name='ALPR Pipeline',
    description='This is the ALPR Pipeline that is meant to be executed on KubeFlow.'
)
def alpr_pipeline(
        image: str = "benjamintanweihao/alpr-kubeflow",
        branch: str = 'master',
        model_name: str = 'ssd_inception_v2_coco'):
    resource_name = 'alpr-pipeline-pvc'

    create_pipeline_volume = create_pipeline_volume_op(resource_name=resource_name)

    git_clone = git_clone_op(branch_or_sha=branch,
                             pvolume=create_pipeline_volume.volume)

    convert_to_tfrecords = convert_to_tfrecords_op(
        image=image,
        pvolume=git_clone.pvolume,
    )

    training_and_eval = train_and_eval_op(image=image,
                                          pvolume=convert_to_tfrecords.pvolume,
                                          model_name=model_name)

    export_save_model_op(image=image,
                         pvolume=training_and_eval.pvolume,
                         model_name=model_name)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'alpr-pipeline.zip')
