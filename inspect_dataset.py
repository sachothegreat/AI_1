import pandas as pd

# Set the correct file path to the 'uploads' directory
file_path = 'uploads/parkinsons.data'  # Adjust this path based on your file's actual location

# Load the data
data = pd.read_csv(file_path)

# Rest of the inspection code
print("Unique 'name' values in the dataset:")
unique_names = data['name'].unique()
for name in unique_names:
    print(name)
