import pandas as pd
import os

df = pd.read_csv('../scrape/psa_sales_190786_20250222_143954.csv')  # Replace with your CSV file path
image_dir = '../scrape/pictures'  # Replace with your image directory path
df['filename'] = df['certNumber'].apply(lambda x: os.path.join(image_dir, f"cert_{x}.jpg"))

print(df.head())

# check if each image exists, and print the filename of images that dont exist
for index, row in df.iterrows():
    if not os.path.exists(row['filename']):
        print(row['filename'])

# for images that dont exist remove it from the df
df = df[df['filename'].apply(lambda x: os.path.exists(x))]

print("\n === CHECKING AGAIN === \n")

# check if each image exists, and print the filename of images that dont exist
for index, row in df.iterrows():
    if not os.path.exists(row['filename']):
        print(row['filename'])