import os

import kfp
from kfp import dsl
from kfp.components import InputPath
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


def convert_to_tfrecords_op(image: str, pvolume: PipelineVolume):
    commands = []
    datasets = ['indian', 'romanian', 'voc']

    commands.append(f'cd {PROJECT_ROOT}')
    commands.append('wget -O DATASETS.tar.xz https://www.dropbox.com/s/qtowh6tq57kd2ss/DATASETS.tar.xz?dl=1')
    commands.append('tar xvf DATASETS.tar.xz')
    commands.append('mkdir TFRECORDS')

    # Set up the tensorflow object detection API
    commands.append(f'cd {PROJECT_ROOT}/MODELS/research')
    commands.append('protoc object_detection/protos/*.proto --python_out=.')
    commands.append('cp object_detection/packages/tf1/setup.py .')
    commands.append('python -m pip install --user .')
    commands.append('cd ../../')

    for d in datasets:
        for split in ['train', 'test']:
            input_dir = os.path.join(PROJECT_ROOT, 'DATASETS', d, split, 'annotations')
            commands.append(
                f'python dataprep/scripts/xml2csv.py -i {input_dir} -o {os.path.join(input_dir, "train_labels.csv")}',
            )

    for d in datasets:
        for split in ['train', 'test']:
            csv_input = os.path.join(PROJECT_ROOT, 'DATASETS', d, split, 'annotations', f'{split}_labels.csv')
            img_path = os.path.join(PROJECT_ROOT, 'DATASETS', d, split, 'images')
            commands.append(
                f'python dataprep/scripts/csv2tfrecords.py --split_name={split} --tfrecord_name={d} '
                f'--label=license-plate --csv_input={csv_input} --output_path TFRECORDS/ --img_path={img_path}'
            )

    for c in commands:
        print(c)

    op = dsl.ContainerOp(
        name='convert to tfrecords',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume},
        file_outputs={'tfrecords': os.path.join(PROJECT_ROOT, 'TFRECORDS')}
    )

    return op


def train_and_eval_op(image: str, pvolume: PipelineVolume, model_name: str, num_train_steps: str):
    model_dir = f'LOGS/{model_name}'

    # Set up the tensorflow object detection API
    commands = [f'cd {PROJECT_ROOT}/MODELS/research',
                'protoc object_detection/protos/*.proto --python_out=.',
                'cp object_detection/packages/tf1/setup.py .',
                'python -m pip install --user .',
                'cd ../../',
                f'cd {PROJECT_ROOT}',
                'wget -O weights.tar.xz https://www.dropbox.com/s/bmdxebtj1cfk9ig/weights.tar.xz?dl=1',
                'tar xvf weights.tar.xz',
                f'export PYTHONPATH=$PYTHONPATH:{os.path.join(PROJECT_ROOT, "MODELS")}',
                f'python train/scripts/model_main.py --model_dir {model_dir} --pipeline_config_path train/model_configs/{model_name}.config --num_train_steps={num_train_steps}',
                f'echo {PROJECT_ROOT}/LOGS/{model_name} > /workspace/model_dir.txt'
                ]

    for c in commands:
        print(c)

    op = dsl.ContainerOp(
        name='train and eval',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume},
        file_outputs={'model_config': f'{PROJECT_ROOT}/train/model_configs/',
                      'model_dir': '/workspace/model_dir.txt'}
    )

    return op


def export_saved_model_op(image: str, pvolume: PipelineVolume, model_name: str, model_version: str,
                          num_train_steps: str):
    checkpoint_prefix = f'LOGS/{model_name}/model.ckpt-{num_train_steps}'

    commands = [
        f'cd {PROJECT_ROOT}/MODELS/research',
        'protoc object_detection/protos/*.proto --python_out=.',
        'cp object_detection/packages/tf1/setup.py .',
        'python -m pip install --user .',
        'cd ../../',
        f'cd {PROJECT_ROOT}',
        f'export PYTHONPATH=$PYTHONPATH:{os.path.join(PROJECT_ROOT, "MODELS")}',
        f'python MODELS/research/object_detection/export_inference_graph.py --input_type image_tensor '
        f'--pipeline_config_path train/model_configs/{model_name}.config '
        f'--trained_checkpoint_prefix {checkpoint_prefix} --output_directory SAVED_MODEL/{model_name}/{model_version}',
        # The model server expects that the saved model is in a versioned folder: e.g. sklearn-iris/4/saved_model.pb
        f'mv SAVED_MODEL/{model_name}/{model_version}/saved_model/saved_model.pb SAVED_MODEL/{model_name}/{model_version}/saved_model.pb',
        f'rm SAVED_MODEL/{model_name}/{model_version}/saved_model/',
        f'echo {PROJECT_ROOT}/SAVED_MODEL/{model_name} > /workspace/model_dir.txt'
    ]

    for c in commands:
        print(c)

    op = dsl.ContainerOp(
        name='export savedmodel',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume},
        file_outputs={'saved_model': f'{PROJECT_ROOT}/SAVED_MODEL',
                      'model_dir': f'/workspace/model_dir.txt'}
    )

    return op


def upload_to_s3_op(
        image: str,
        pvolume: PipelineVolume,
        model_dir: InputPath(str),
        export_bucket: str,
        model_name: str):
    commands = [
        f'cd {PROJECT_ROOT}',
        f'python export/upload_to_s3.py --model_dir={model_dir} '
        f'--export_bucket={export_bucket} '
        f'--model_name={model_name} '
    ]

    for c in commands:
        print(c)

    op = dsl.ContainerOp(
        name='upload model to MinIO',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


# A couple of things needs to happen before this would work
# 1. Within this namespace, you would need to create two things:
#
# a) A Secret that would contain the MinIO credentials
# b) A ServiceAccount that points to this Secret.
#
#    kubectl create -f serving/minio_s3_secrets.yaml
#
# The Serving Op will then reference this ServiceAccount in order to
# access the MinIO credentials.

def serving_op(export_bucket: str, model_name: str, model_dns_prefix: str, model_version: str):
    kfserving_op = kfp.components.load_component_from_url(
        'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/kfserving/component.yaml')

    op = kfserving_op(
        action="create",
        model_uri=f"s3://{export_bucket}/{model_name}/{model_version}",
        model_name=model_dns_prefix,  # this should be DNS friendly
        namespace='kubeflow-user',
        framework="tensorflow",
        service_account="sa",
        min_replicas='1',
        max_replicas='1',
    )

    return op


@dsl.pipeline(
    name='ALPR Pipeline',
    description='This is the ALPR Pipeline that is meant to be executed on KubeFlow.'
)
def alpr_pipeline(
        image: str = "benjamintanweihao/alpr-kubeflow",
        branch: str = 'master',
        model_name: str = 'ssd_inception_v2_coco',
        model_dns_prefix: str = 'ssd-inception-v2',
        num_train_steps: str = '20000',
        export_bucket: str = 'servedmodels',
        model_version: str = '1'
):
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
                                          model_name=model_name,
                                          num_train_steps=num_train_steps)

    export_saved_model = export_saved_model_op(image=image,
                                               pvolume=training_and_eval.pvolume,
                                               model_name=model_name,
                                               model_version=model_version,
                                               num_train_steps=num_train_steps)

    upload_to_s3 = upload_to_s3_op(image=image,
                                   pvolume=export_saved_model.pvolume,
                                   model_dir=export_saved_model.outputs['model_dir'],
                                   export_bucket=export_bucket,
                                   model_name=model_name)

    _ = serving_op(
        export_bucket=export_bucket,
        model_name=model_name,
        model_dns_prefix=model_dns_prefix,
        model_version=model_version).after(upload_to_s3)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'alpr-pipeline.zip')
