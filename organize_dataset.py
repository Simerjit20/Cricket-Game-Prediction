
import os
import shutil

base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "dataset")

os.makedirs(dataset_path, exist_ok=True)

items = [
    "COMMENTARY_INTL_MATCH",
    "INTERNATIONAL_MATCH.csv"
]

for item in items:
    source = os.path.join(base_path, item)
    destination = os.path.join(dataset_path, item)

    if os.path.exists(source):
        shutil.move(source, destination)
        print(item, "moved successfully!")

    elif os.path.exists(destination):
        print(item, "already inside dataset!")

    else:
        print(item, "not found!")

print("Dataset organization completed!")
