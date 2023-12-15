import os
import json

bucket_path = "s3://amazon-reviews-ea/landing"
file = open("kaggle_filenames.json")
filenames = json.load(file)

for i, filename in enumerate(filenames):
    os.system(
        f"kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset -f {filename}"
    )
    print("Unzipping {filename}.zip . . .")
    os.system(f"unzip {filename}.zip")
    print("Uploading to S3 Bucket . . .")
    os.system(f"aws s3 cp {filename} {bucket_path}/{filename}")
    print(f"Removing {filename} from EC2 . . .")
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(f"{filename}.zip"):
        os.remove(f"{filename}.zip")
    print(f"Completed {i+1}/{len(filenames)}!\n")
