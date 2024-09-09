import splitfolders

# Path to the raw dataset with subfolders for each class
input_folder = 'raw_dataset'

# Output folder where the split dataset will be saved
output_folder = 'dataset'

# Split the dataset into training (80%) and validation (20%) sets
# This will automatically create the "train" and "validation" folders
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .2))

print("Dataset has been organized into 'train/' and 'validation/' folders.")
