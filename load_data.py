import tensorflow as tf

# Path to your dataset folders
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image settings
img_height = 128
img_width = 128
batch_size = 32

# Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"   # since we have 2 classes: REAL vs FAKE
)

# Load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

class_names = train_ds.class_names

# Normalize pixel values (0-255 -> 0-1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Speed up data loading during training
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# Print class names to confirm labels loaded correctly
print("Classes found:", class_names)

# Look at one batch to confirm shapes
for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)