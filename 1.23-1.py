from os import listdir
from os.path import join
from numpy import asarray, save
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define location of dataset
folder = 'train/'
photos, labels = list(), list()

# Enumerate files in the directory
for file in listdir(folder):
    # Filter out non-image files
    if not file.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Skipped non-image file: {file}")
        continue
    # Determine class
    output = 0.0
    if file.startswith('cat'):
        output = 1.0
    try:
        # Load image
        photo = load_img(join(folder, file), target_size=(200, 200))
        # Convert to numpy array
        photo = img_to_array(photo)
        # Store
        photos.append(photo)
        labels.append(output)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Convert to numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)

# Save the reshaped photos
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)