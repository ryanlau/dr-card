import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import os

df = pd.read_csv('../scrape/psa_sales_190786_20250222_143954.csv')  # Replace with your CSV file path
image_dir = '../scrape/pictures'  # Replace with your image directory path
df['filename'] = df['certNumber'].apply(lambda x: os.path.join(image_dir, f"cert_{x}.jpg"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        

print("Unique grades in full dataset:", df['grade'].unique())
print("Number of unique grades in full dataset:", df['grade'].nunique())

grade_counts = df['grade'].value_counts()

# also add percent of total to grade_counts as a new column
grade_counts = grade_counts.reset_index()
grade_counts.columns = ['grade', 'count']
grade_counts['percent'] = grade_counts['count'] / len(df) * 100

print(grade_counts)

# Split into training and validation sets
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Encode the grade labels (e.g., 'PSA1' to 'PSA10') into integers
le = LabelEncoder()
le.fit(df['grade'])
train_df['label'] = le.transform(train_df['grade'])
val_df['label'] = le.transform(val_df['grade'])

# Optional: Compute class weights to handle imbalance
classes = np.unique(train_df['grade'])
class_weights = compute_class_weight('balanced', classes=classes, y=train_df['grade'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


# print all output of above to understand the data
print("Unique grades in full dataset:", df['grade'].unique())
print("Number of unique grades in full dataset:", df['grade'].nunique())
print("Unique grades in train_df:", train_df['grade'].unique())
print("Number of unique grades in train_df:", train_df['grade'].nunique())
print("Unique grades in val_df:", val_df['grade'].unique())
print("Number of unique grades in val_df:", val_df['grade'].nunique())
# print("Unique grades in full dataset:", df['grade'].unique())
# print("Number of unique grades in full dataset:", df['grade'].nunique())

print(class_weights)
print(class_weights_tensor)

# Define the desired grades
# desired_grades = [1.0, 2.0, 3.0, 4.0, 5.0, 
                #   6.0, 7.0, 8.0, 9.0, 10.0]

# print(df['grade'])

# df = df[df['grade'].isin(desired_grades)]

# print("Unique grades in full dataset:", df['grade'].unique())
# print("Number of unique grades in full dataset:", df['grade'].nunique())





# print("Unique grades in train_df:", train_df['grade'].unique())
# print("Number of unique grades in train_df:", train_df['grade'].nunique())