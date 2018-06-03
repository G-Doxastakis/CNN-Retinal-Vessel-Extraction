import driveDataset

from keras.models import Sequential
from keras.layers import Conv2D,   Reshape, AveragePooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_train, Y_train = driveDataset.loadImages(mode='training')
X_test, Y_test = driveDataset.loadImages(mode='test')
print(X_train.shape, '->', Y_train.shape)
print(X_test.shape, '->', Y_test.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=7, padding='same', activation='tanh', input_shape=(584, 565, 3)))
model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='tanh'))
model.add(Conv2D(filters=1, kernel_size=3, dilation_rate=2, padding='same', activation='sigmoid'))
model.add(Reshape((584, 565)))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))

print(model.summary())

stop = EarlyStopping(monitor='loss', patience=3, min_delta=0.0005)
log = TensorBoard()
model.fit(x=X_train, y=Y_train, batch_size=10, shuffle=True, epochs=300, callbacks=[stop, log])
model.save('eye_vessel.h5')

print('Predicting Test...')
Y_pred = model.predict(X_test)
print('Saving Images...')
driveDataset.saveImages(Y_test, Y_pred)
print('Calculating Loss...')
score = model.evaluate(x=X_test, y=Y_test)
print("Loss: %.2f" % score)
