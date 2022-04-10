import kfp.components as comp


def export_saved_model_op(image: str):

    return comp.load_component_from_text(f"""
    name: Export SavedModel
    description: Export SavedModel
    inputs:
    - {{name: project_workspace, type: Directory}}
    - {{name: model_name, type: String}}
    - {{name: num_train_steps, type: String}}
    - {{name: model_version, type: String, default: '1'}}
    - {{name: model_dir, type: OutputPath}}
    outputs:
    - {{name: saved_model_dir, type: OutputPath}}
    implementation:
      container:
        image: {image} 
        command:
        - /bin/bash
        - -c 
        - $0/scripts/export.sh $0 $1 $2 $3 $5 $4
        args:
        - inputPath: project_workspace # $1
        - inputValue: model_name       # $2
        - inputValue: num_train_steps  # $3
        - inputPath: model_dir         # $4
        - outputPath: saved_model_dir  # $5
        - inputValue: model_version    # $6
    """)


def upload_to_s3_op(image: str):
    return comp.load_component_from_text(f"""
name: Upload SavedModel to S3
description: Upload SavedModel S3
inputs:
- {{name: project_workspace, type: Directory}}
- {{name: model_dir, type: OutputPath}}
- {{name: model_name, type: String}}
- {{name: export_bucket, type: String, default: 'servedmodels'}}
implementation:
  container:
    image: {image} 
    command:
    - /bin/bash
    - -c 
    - $0/scripts/upload.sh $0 $1 $2 $3
    args:
    - inputPath: project_workspace # $1
    - inputPath: model_dir         # $2
    - inputValue: model_name       # $2
    - inputValue: export_bucket    # $3
""")
