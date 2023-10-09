import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2

# Загрузка данных
train = pd.read_csv('C:/Users/frien/source/ExampleContest/CosmiticJunWithPower/train.csv')

# Предобработка данных
X = train.drop('target', axis=1).values
y = train['target'].values

# Разделение данных на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Создание модели
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5),  # dropout для регуляризации и предотвращения переобучения
    keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Ранняя остановка
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Обучение модели
history = model.fit(
    X_train, y_train, 
    epochs=1000,  # высокое значение, так как у нас есть ранняя остановка
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]  # добавление ранней остановки
)

# Оценка модели
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}")

# Сохранение модели
model.save('trained_model.h5')
