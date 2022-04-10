import kfp
import kfp.components as comp
from kfp import dsl
from kfp.v2.components.experimental.base_component import BaseComponent
from kfp.v2.components.experimental.pipeline_task import PipelineTask

from serving.component import serving_op

git_clone_op: BaseComponent = comp.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/contrib/git/clone/component.yaml')

image: str = "benjamintanweihao/alpr-kubeflow"
serving_step: BaseComponent = serving_op(image=image)


@dsl.pipeline(
    name='ALPR Pipeline',
    description='This is the ALPR Pipeline that is meant to be executed on KubeFlow.'
)
def serving_pipeline(
        model_dns_prefix: str = 'ssd-inception-v2',
):
    git_clone_step: PipelineTask = git_clone_op('https://github.com/benjamintanweihao/alpr-with-kubeflow.git', '1.5')

    project_workspace = git_clone_step.output

    _ = serving_step(project_workspace=project_workspace,
                     model_dns_prefix=model_dns_prefix)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(serving_pipeline, 'serving_pipeline.yaml')
