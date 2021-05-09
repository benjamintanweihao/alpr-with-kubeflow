import kfp
from kfp import dsl

from kubeflow.components import (create_pipeline_volume_op,
                                 git_clone_op,
                                 convert_to_tfrecords_op,
                                 train_and_eval_op,
                                 export_saved_model_op,
                                 upload_to_s3_op,
                                 serving_op)


@dsl.pipeline(
    name='ALPR Pipeline',
    description='This is the ALPR Pipeline that is meant to be executed on KubeFlow.'
)
def alpr_pipeline(
        image: str = "benjamintanweihao/alpr-kubeflow",
        branch: str = 'master',
        model_name: str = 'ssd_inception_v2_coco',
        model_dns_prefix: str = 'ssd-inception-v2',
        num_train_steps: str = '20',  # 20000 for full training
        export_bucket: str = 'servedmodels',
        model_version: str = '1'
):
    resource_name = 'alpr-pipeline-pvc'

    create_pipeline_volume = create_pipeline_volume_op(resource_name=resource_name)

    git_clone = git_clone_op(branch_or_sha=branch,
                             pvolume=create_pipeline_volume.volume)

    convert_to_tfrecords = convert_to_tfrecords_op(
        image=image,
        pvolume=git_clone.pvolume)

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

    _ = serving_op(image=image,
                   model_dns_prefix=model_dns_prefix,
                   pvolume=upload_to_s3)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(alpr_pipeline, 'alpr-pipeline.zip')
