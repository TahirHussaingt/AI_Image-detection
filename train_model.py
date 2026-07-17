import tensorflow as tf
from keras import layers, models

# ---- Image settings ----
img_height = 224
img_width = 224
batch_size = 32

train_dir = "dataset_v2/train"
test_dir = "dataset_v2/test"

# ---- Load data ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

class_names = train_ds.class_names
print("Classes found:", class_names)

# ---- Prefetch for speed (NOTE: no manual rescaling needed - MobileNetV2 handles it) ----
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ---- Load pretrained MobileNetV2 (without its final classification layer) ----
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,       # we don't want its original 1000-category output
    weights='imagenet'       # use its pretrained knowledge
)

# Freeze the pretrained layers - we don't want to change what it already knows
base_model.trainable = False

# ---- Build our full model ----
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

# MobileNetV2 expects pixels in range [-1, 1], this layer handles that automatically
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# ---- Compile ----
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---- Train ----
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)

# ---- Save ----
model.save("image_detector_model.keras")
print("Model saved as image_detector_model.keras")