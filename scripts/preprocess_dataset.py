import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# --- Configuration ---
# Adjust these paths based on where your CSV files ARE or WILL BE
RAW_DATA_DIR = "data/raw" 
TRAIN_CSV = os.path.join(RAW_DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(RAW_DATA_DIR, "sign_mnist_test.csv")

# Output directories for ImageFolder structure
DATA_DIR = "data/processed/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Validation split percentage (e.g., 20% of training data for validation)
VAL_SPLIT_RATIO = 0.2 
RANDOM_SEED = 42 # for reproducibility

TARGET_CLASSES = [0, 8, 11, 14, 21] # Keep only labels for A (fist), I (pinkie), L (thumb index), O (grip), V (peace)
TARGET_CLASS_STR = [str(c) for c in TARGET_CLASSES] # String versions for paths

# --- Function to process CSV and save images ---
def csv_to_image_folders(csv_path, base_output_dir, target_labels):
    """Reads SignMNIST CSV, filters for target labels, and saves images."""
    print(f"Processing {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV files are in the RAW_DATA_DIR.")
        return None, None # Return None to indicate failure
    
    # *** Filter the DataFrame FIRST ***
    df_filtered = df[df['label'].isin(target_labels)]
    print(f"  Filtered down to {len(df_filtered)} samples for classes {target_labels}")

    labels = df_filtered['label'].values
    # Drop the label column to get pixel data
    pixels = df_filtered.drop('label', axis=1).values 

    # Create base output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    image_paths = []
    image_labels = []

    # Save images
    num_images = len(labels)
    for i in range(num_images):
        label = labels[i]
        # Reshape the 784 pixels into a 28x28 array
        img_array = pixels[i].reshape(28, 28).astype(np.uint8) 
        img = Image.fromarray(img_array)

        # Create class directory if it doesn't exist
        class_dir = os.path.join(base_output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        # Define image path and save
        img_filename = f"img_{i}.png"
        img_path = os.path.join(class_dir, img_filename)
        img.save(img_path)

        # Store path and label for potential splitting later
        image_paths.append(img_path)
        image_labels.append(label)

        # Optional: Print progress
        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{num_images} images...")

    print(f"Finished saving images to {base_output_dir}")
    return image_paths, image_labels

# --- Main Execution ---

# 1. Ensure raw data directory exists (or create it and place CSVs there)
if not os.path.isdir(RAW_DATA_DIR):
    print(f"Creating raw data directory: {RAW_DATA_DIR}")
    print(f"Please place sign_mnist_train.csv and sign_mnist_test.csv inside {RAW_DATA_DIR}")
    os.makedirs(RAW_DATA_DIR)
    # Exit if CSVs aren't there yet
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
         print("Exiting. Place CSV files in data_raw/ and rerun.")
         exit()


# 2. Process the test set first (no splitting needed)
print("\n--- Processing Test Set ---")
test_paths, _ = csv_to_image_folders(TEST_CSV, TEST_DIR, TARGET_CLASSES)
if test_paths is None:
    exit() # Stop if test CSV processing failed

# 3. Process the training set (to a temporary directory initially)
print("\n--- Processing Training Set ---")
TEMP_TRAIN_DIR = os.path.join(DATA_DIR, "temp_train") # Temporary holding place
train_val_paths, train_val_labels = csv_to_image_folders(TRAIN_CSV, TEMP_TRAIN_DIR, TARGET_CLASSES)
if train_val_paths is None:
    exit() # Stop if train CSV processing failed


# 4. Split training data into train and validation sets
print("\n--- Splitting Training Data into Train/Validation ---")
# Use stratify to keep class proportions similar in train and val sets
train_paths, val_paths, _, _ = train_test_split(
    train_val_paths, 
    train_val_labels, 
    test_size=VAL_SPLIT_RATIO, 
    random_state=RANDOM_SEED,
    stratify=train_val_labels # Important for classification
)

# 5. Move validation files to the VAL_DIR
os.makedirs(VAL_DIR, exist_ok=True)
print(f"Moving {len(val_paths)} files to {VAL_DIR}...")
for file_path in val_paths:
    # Extract label and filename from the path
    parts = file_path.split(os.sep) # e.g., ['data', 'temp_train', '5', 'img_123.png']
    label = parts[-2]
    filename = parts[-1]
    
    # Create destination class directory
    dest_class_dir = os.path.join(VAL_DIR, label)
    os.makedirs(dest_class_dir, exist_ok=True)
    
    # Define destination path
    dest_path = os.path.join(dest_class_dir, filename)
    
    # Move the file
    try:
        shutil.move(file_path, dest_path)
    except Exception as e:
        print(f"Error moving {file_path} to {dest_path}: {e}")

# 6. Rename TEMP_TRAIN_DIR to TRAIN_DIR (now contains only training files)
print(f"Renaming {TEMP_TRAIN_DIR} to {TRAIN_DIR}...")
try:
    # Remove existing TRAIN_DIR if it exists to avoid errors on rerun
    if os.path.exists(TRAIN_DIR):
         shutil.rmtree(TRAIN_DIR)
    os.rename(TEMP_TRAIN_DIR, TRAIN_DIR)
except Exception as e:
    print(f"Error renaming directory: {e}")
    print(f"Please manually rename {TEMP_TRAIN_DIR} to {TRAIN_DIR} if necessary.")


print("\n--- Dataset Preparation Complete ---")
print(f"Training images: {len(train_paths)} (in {TRAIN_DIR})")
print(f"Validation images: {len(val_paths)} (in {VAL_DIR})")
print(f"Test images: {len(test_paths)} (in {TEST_DIR})")