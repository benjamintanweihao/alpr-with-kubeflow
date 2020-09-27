import kfp
from kfp import dsl


def mnist_op(image: str):
    return dsl.ContainerOp(
        name='MNIST',
        image=image,
        command=['sh'],
        arguments=['-c', 'python mnist.py'],
        container_kwargs={'image_pull_policy': 'IfNotPresent'}
    )


@dsl.pipeline(
    name='Hello World Pipeline',
    description='This is a single component pipeline'
)
def hello_world_pipeline(image: str = 'benjamintanweihao/hello-world:latest'):
    mnist_op(image=image)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(hello_world_pipeline, 'hello-world-pipeline.zip')
