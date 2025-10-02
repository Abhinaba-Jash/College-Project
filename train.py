from keras.preprocessing.image import ImageDataGenerator
from mesonet import Meso4

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Model
model = Meso4()
model.summary()

# Training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save model
model.save("mesonet_model.h5")
print("✅ Model saved as mesonet_model.h5")
