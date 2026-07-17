import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("image_detector_model.keras")

img_height = 224
img_width = 224
batch_size = 32

test_dir = "dataset_v2/test"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

# Get all predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().flatten())
    y_pred.extend((preds.flatten() > 0.5).astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Overall accuracy
accuracy = np.mean(y_true == y_pred)
print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")

# Breakdown: how many of each class did it predict?
print(f"\nActual REAL count: {np.sum(y_true == 1)}")
print(f"Actual FAKE count: {np.sum(y_true == 0)}")
print(f"Predicted REAL count: {np.sum(y_pred == 1)}")
print(f"Predicted FAKE count: {np.sum(y_pred == 0)}")

# Per-class accuracy
real_mask = y_true == 1
fake_mask = y_true == 0
print(f"\nAccuracy on REAL images: {np.mean(y_pred[real_mask] == y_true[real_mask])*100:.2f}%")
print(f"Accuracy on FAKE images: {np.mean(y_pred[fake_mask] == y_true[fake_mask])*100:.2f}%")