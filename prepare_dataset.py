import pandas as pd
import os
import shutil
import random

# ---- Settings ----
csv_path = "train.csv"
source_folder = "train_data"          # where all mixed images currently live
output_folder = "dataset_v2"          # new clean folder we're creating

samples_per_class = 7500              # how many REAL and FAKE images to pull
test_split = 0.15                     # 15% held back for testing

# ---- Read the CSV ----
df = pd.read_csv(csv_path)
print("Total rows in CSV:", len(df))
print(df.head())

# ---- Separate by label ----
real_df = df[df['label'] == 0]
fake_df = df[df['label'] == 1]

print("Total REAL available:", len(real_df))
print("Total FAKE available:", len(fake_df))

# ---- Randomly sample ----
real_sample = real_df.sample(n=samples_per_class, random_state=42)
fake_sample = fake_df.sample(n=samples_per_class, random_state=42)

# ---- Split each into train/test ----
def split_train_test(data, split_ratio):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    split_idx = int(len(data) * (1 - split_ratio))
    return data[:split_idx], data[split_idx:]

real_train, real_test = split_train_test(real_sample, test_split)
fake_train, fake_test = split_train_test(fake_sample, test_split)

# ---- Create folders ----
folders = [
    f"{output_folder}/train/REAL",
    f"{output_folder}/train/FAKE",
    f"{output_folder}/test/REAL",
    f"{output_folder}/test/FAKE",
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ---- Copy files ----
def copy_files(data, destination_folder):
    copied = 0
    missing = 0
    for _, row in data.iterrows():
        # file_name column already includes "train_data/" prefix
        src_path = row['file_name']
        filename = os.path.basename(src_path)
        dst_path = os.path.join(destination_folder, filename)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copied += 1
        else:
            missing += 1
    return copied, missing

print("Copying REAL train images...")
c, m = copy_files(real_train, f"{output_folder}/train/REAL")
print(f"Copied: {c}, Missing: {m}")

print("Copying FAKE train images...")
c, m = copy_files(fake_train, f"{output_folder}/train/FAKE")
print(f"Copied: {c}, Missing: {m}")

print("Copying REAL test images...")
c, m = copy_files(real_test, f"{output_folder}/test/REAL")
print(f"Copied: {c}, Missing: {m}")

print("Copying FAKE test images...")
c, m = copy_files(fake_test, f"{output_folder}/test/FAKE")
print(f"Copied: {c}, Missing: {m}")

print("\nDone! Dataset structure created at:", output_folder)