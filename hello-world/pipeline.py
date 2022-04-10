import kfp
import kfp.components as comp


mnist_op = comp.load_component_from_file('component.yaml')


def hello_world_pipeline():
    _ = mnist_op()


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(hello_world_pipeline, 'hello-world-pipeline.yaml')
