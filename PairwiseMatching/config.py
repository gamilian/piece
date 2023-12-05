import os

dataset = "MIT_ex"
dataset_path = os.path.join("/work/csl/code/piece/dataset/raw_dataset", dataset)


num_fragment = 0
for filename in os.listdir(dataset_path):
        if filename.startswith("fragment") and filename.endswith(".png"):
            num_fragment += 1

with open (os.path.join(dataset_path, "bg_color.txt"), "r") as f:
    bg_color = [int(x) for x in f.readline().split()][::-1]

# crop_image_name = f"szp.png"

log_path = os.path.join("log", f"{dataset}.log")

alignments_file = os.path.join(dataset_path, 'alignments.txt')

num_processes = 32