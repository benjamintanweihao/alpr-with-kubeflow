import kfp.components as comp

def convert_to_tfrecords_op(image: str):
    return comp.load_component_from_text(f"""
  name: Create TFRecords
  description: Create TFRecords
  inputs:
  - {{name: project_workspace, type: Directory}}
  outputs:
  - {{name: data_dir, type: OutputPath}}
  implementation:
    container:
      image: {image} 
      command:
      - /bin/bash
      - -c
      - $0/scripts/dataprep.sh $0 $1
      args:
      - inputPath: project_workspace
      - outputPath: data_dir
  """)
