import kfp
import kfp.components as comp
from kfp import dsl
from kfp.v2.components.experimental.base_component import BaseComponent
from kfp.v2.components.experimental.pipeline_task import PipelineTask

from dataprep.component import convert_to_tfrecords_op
from export.component import export_saved_model_op, upload_to_s3_op
from serving.component import serving_op
from train.component import train_and_eval_op

git_clone_op: BaseComponent = comp.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/contrib/git/clone/component.yaml')

image: str = "benjamintanweihao/alpr-kubeflow"
convert_to_tfrecords_step: BaseComponent = convert_to_tfrecords_op(image=image)
train_and_eval_step: BaseComponent = train_and_eval_op(image=image)
export_saved_model_step: BaseComponent = export_saved_model_op(image=image)
upload_to_s3_step: BaseComponent = upload_to_s3_op(image=image)
serving_step: BaseComponent = serving_op(image=image)


@dsl.pipeline(
    name='ALPR Pipeline',
    description='This is the ALPR Pipeline that is meant to be executed on KubeFlow.'
)
def pipeline(
        model_name: str = 'ssd_inception_v2_coco',
        num_train_steps: str = '1',  # 20000 steps for full training
        model_dns_prefix: str = 'ssd-inception-v2',
):
    git_clone_step: PipelineTask = git_clone_op('https://github.com/benjamintanweihao/alpr-with-kubeflow.git', '1.5')

    project_workspace = git_clone_step.output

    convert_to_tfrecords = convert_to_tfrecords_step(project_workspace=project_workspace)

    train_and_eval = train_and_eval_step(project_workspace=project_workspace,
                                         model_name=model_name,
                                         num_train_steps=num_train_steps,
                                         data_dir=convert_to_tfrecords.outputs['data_dir'])

    export_saved_model = export_saved_model_step(project_workspace=project_workspace,
                                                 model_name=model_name,
                                                 num_train_steps=num_train_steps,
                                                 model_dir=train_and_eval.outputs['model_dir'])

    upload_to_s3 = upload_to_s3_step(project_workspace=project_workspace,
                                     model_dir=export_saved_model.outputs['saved_model_dir'],
                                     model_name=model_name)

    _ = serving_step(project_workspace=project_workspace,
                     model_dns_prefix=model_dns_prefix).after(upload_to_s3)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, 'pipeline.yaml')
