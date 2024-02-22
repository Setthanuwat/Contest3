from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


BATCH_SIZE = 20
IMAGE_SIZE = (72,108 )

datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=[0.9,1.1],
            shear_range=1,
            zoom_range=0.05,
            rotation_range=10,
            width_shift_range=0.03,
            height_shift_range=0.03,
            vertical_flip=True,
            horizontal_flip=True)

# Download dataset form https://drive.google.com/file/d/1jwa16s2nZIQywKMdRkpRvdDifxGDxC3I/view?usp=sharing
dataframe = pd.read_csv('contest_foodweight/fried_noodles_dataset.csv', delimiter=',', header=0)

print(dataframe)
print(dataframe.loc[0:1])

datagen_noaug = ImageDataGenerator(rescale=1./255)
dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)


train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:1400],
    directory='contest_foodweight/images',
    x_col='filename',
    y_col=['meat', 'veggie', 'noodle'],
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, 
    class_mode='other')

validation_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[1401:1500],
    directory='contest_foodweight/images',
    x_col='filename',
    y_col=['meat', 'veggie', 'noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[1501:1855],
    directory='contest_foodweight/images',
    x_col='filename',
    y_col=['meat', 'veggie', 'noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

inputIm = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3,))
conv1 = Conv2D(16, 5, activation='relu')(inputIm)
pool1 = MaxPool2D()(conv1)
conv1 = Conv2D(16, 5, activation='relu')(inputIm)
pool1 = MaxPool2D()(conv1)
flat = Flatten()(pool1)
dense1 = Dense(16, activation='relu')(flat)
#dense1 = Dropout(0.5)(dense1)
predictedWeights = Dense(3, activation='relu')(dense1)  # Output for 'meat', 'veggie', 'noodle'

model = Model(inputs=inputIm, outputs=predictedWeights)

model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mean_absolute_error'])

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('mean_absolute_error'))
        self.val_losses.append(logs.get('val_mean_absolute_error'))

        plt.clf()
        plt.plot(self.x, self.losses, label='mean_absolute_error')
        plt.plot(self.x, self.val_losses, label='val_mean_absolute_error')
        plt.legend()
        plt.pause(1)


checkpoint = ModelCheckpoint('fried_noodles_best9.h5', verbose=1, monitor='val_mean_absolute_error', save_best_only=True, mode='min')
plot_losses = PlotLosses()

# Train Model
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=200,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, plot_losses])

# Test
model = load_model('fried_noodles_best9.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n', score)
 
test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers=1,
    use_multiprocessing=False)

# Denormalize the predicted weights
print('prediction:\n', predict)
# Create a DataFrame for the results

result_df = pd.DataFrame({
    'filename': test_generator.filenames,
    'meat': predict[:, 0],
    'veggie': predict[:, 1],
    'noodle': predict[:, 2]
})

# Save the DataFrame to a CSV file
result_df.to_csv('result.csv', index=False)

# Display the first few rows of the result DataFrame
print(result_df.head())
print("test data_meat",test_generator.labels[:, 0])
print("test data_veggie",test_generator.labels[:, 1])
print("test data_noodle",test_generator.labels[:, 2])


# Calculate Mean Absolute Error (MAE) for each class
mae_meat = np.mean(np.abs(predict[:, 0] - test_generator.labels[:, 0]))
mae_veggie = np.mean(np.abs(predict[:, 1] - test_generator.labels[:, 1]))
mae_noodle = np.mean(np.abs(predict[:, 2] - test_generator.labels[:, 2]))

print('MAE for meat:', mae_meat, 'grams')
print('MAE for veggie:', mae_veggie, 'grams')
print('MAE for noodle:', mae_noodle, 'grams')

# Total MAE
total_mae = np.mean([mae_meat, mae_veggie, mae_noodle])
print('Total MAE:', total_mae, 'grams')
