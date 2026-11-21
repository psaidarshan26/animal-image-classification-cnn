import tensorflow as tf
from pathlib import Path
import json

DATA_DIR = Path("data/animals")
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 5

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Download the 90 Animal Image Dataset and arrange images as "
        "data/animals/<animal_name>/<image_files>."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Number of animal classes:", num_classes)
print("Animal classes:", ", ".join(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 255, input_shape=(*IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"\nValidation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

model.save("animal_classifier.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
print("Saved model as animal_classifier.keras")
print("Saved labels as class_names.json")
