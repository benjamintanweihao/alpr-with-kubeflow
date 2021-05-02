import os
import argparse
import boto3
from botocore.client import Config


def main():
    parser = argparse.ArgumentParser(description="upload model to S3")
    parser.add_argument("-d",
                        "--model_dir",
                        help="path of the model to be exported",
                        type=str)
    parser.add_argument("-b",
                        "--export_bucket",
                        help="name of bucket to export to",
                        type=str)
    parser.add_argument("-n",
                        "--model_name",
                        help="name of the model",
                        type=str)
    parser.add_argument("-v",
                        "--model_version",
                        help="version of the model",
                        type=int)

    args = parser.parse_args()

    model_dir = args.model_dir
    model_name = args.model_name
    model_version = args.model_version
    export_bucket = args.export_bucket

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
    )

    # Create export bucket if it does not yet exist
    response = s3.list_buckets()
    export_bucket_exists = False

    for bucket in response["Buckets"]:
        if bucket["Name"] == export_bucket:
            export_bucket_exists = True

    if not export_bucket_exists:
        s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

    # Save model files to S3
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, model_dir)

            s3.upload_file(
                local_path,
                export_bucket,
                f"{model_name}/{model_version}/{s3_path}",
                ExtraArgs={"ACL": "public-read"},
            )

    response = s3.list_objects(Bucket=export_bucket)
    print(f"All objects in {export_bucket}:")
    for file in response["Contents"]:
        print("{}/{}".format(export_bucket, file["Key"]))


if __name__ == '__main__':
    main()
