import os
import argparse
import boto3


DEFAULT_PATH = os.path.abspath('../../data')
DEFAULT_BUCKETNAME = os.getenv('BUCKET_NAME', 'mlflow-enkidupal-experiments')



def upload_directory(path, bucketname):
    s3C = boto3.client('s3')
    for root,dirs,files in os.walk(path):
        for file in files:
            s3C.upload_file(os.path.join(root, file), bucketname, file)


def upload_directory_better(path, bucketname, destination):
    client = boto3.client('s3')
    for root, dirs, files in os.walk(path):

        for filename in files:

            # construct the full local path
            local_path = os.path.join(root, filename)

            relative_path = os.path.relpath(local_path, path)
            s3_path = os.path.join(destination, relative_path)

            # relative_path = os.path.relpath(os.path.join(root, filename))

            print ('Searching "%s" in "%s"' % (s3_path, bucketname) )
            try:
                client.head_object(Bucket=bucketname, Key=s3_path)
                print ("Path found on S3! Skipping %s..." % s3_path )

                # try:
                # client.delete_object(Bucket=bucket, Key=s3_path)
                # except:
                # print "Unable to delete %s..." % s3_path
            except:
                print ("Uploading %s..." % s3_path)
                client.upload_file(local_path, bucketname, s3_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PATH, help="Path to upload")
    parser.add_argument("--bucketname", default=DEFAULT_BUCKETNAME, help="Bucket name to upload data to")
    parser.add_argument("--destination", default='data', help="Destination path to upload in s3")

    args = parser.parse_args()

    upload_directory_better(path=args.path, bucketname=args.bucketname, destination=args.destination)