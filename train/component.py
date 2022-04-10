import kfp.components as comp


def train_and_eval_op(image: str):

    return comp.load_component_from_text(f"""
    name: Train and Eval
    description: Train and Eval
    inputs:
    - {{name: project_workspace, type: Directory}}
    - {{name: model_name, type: String}}
    - {{name: num_train_steps, type: String}}
    - {{name: data_dir, type: OutputPath}}
    outputs:
    - {{name: model_dir, type: OutputPath}}
    implementation:
      container:
        image: {image} 
        command:
        - /bin/bash
        - -c 
        - $0/scripts/train.sh $0 $1 $2 $3 $4
        args:
        - inputPath: project_workspace # $1
        - inputValue: model_name       # $2
        - inputValue: num_train_steps  # $3
        - inputPath: data_dir          # $4
        - outputPath: model_dir        # $5
    """)
