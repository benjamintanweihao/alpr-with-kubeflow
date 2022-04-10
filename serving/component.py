import kfp.components as comp


def serving_op(image: str):
    return comp.load_component_from_text(f"""
    name: Serve Model
    description: Serve Model
    inputs:
    - {{name: project_workspace, type: Directory}}
    - {{name: model_dns_prefix, type: String}}
    implementation:
      container:
        image: {image} 
        command:
        - /bin/bash
        - -c 
        - $0/scripts/serve.sh $0 $1
        args:
        - inputPath: project_workspace # $1
        - inputValue: model_dns_prefix # $2
    """)
