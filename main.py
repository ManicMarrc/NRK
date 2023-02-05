import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets

import sys
import pygame

from typing import Tuple

def get_model() -> keras.Model:
  try:
    return keras.models.load_model('model.h5')
  except (ImportError, IOError):
    inputs = keras.Input(shape=(28, 28, 1))
    layer = layers.Conv2D(64, kernel_size=(2, 2), activation='relu')(inputs)
    layer = layers.MaxPooling2D(pool_size=(1, 1))(layer)
    layer = layers.Conv2D(64, kernel_size=(4, 4), activation='relu')(layer)
    layer = layers.MaxPooling2D(pool_size=(3, 3))(layer)
    layer = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = layers.Flatten()(layer)
    layer = layers.Dropout(0.55)(layer)
    outputs = layers.Dense(10, activation='softmax')(layer)

    model = keras.Model(inputs, outputs)

    data: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    x_train = np.expand_dims(x_train.astype('float32') / 255, axis=-1)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    x_test = np.expand_dims(x_test.astype('float32') / 255, axis=-1)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=30, validation_split=0.01)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'{loss = :.3f}, {acc = :.3f}')

    model.save('model.h5')
    return model

def main():
  model = get_model()

  img_upscale = int(600 / 28)

  pygame.init()
  screen = pygame.display.set_mode((800, 28 * img_upscale))
  font = pygame.font.Font(None, 64)

  img = np.zeros((28, 28, 1))
  predicted = None

  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and pygame.key.get_pressed()[pygame.K_ESCAPE]): sys.exit()
      if event.type == pygame.KEYDOWN and pygame.key.get_pressed()[pygame.K_DELETE]:
        img = np.zeros_like(img)
        predicted = None
      if event.type == pygame.KEYDOWN and pygame.key.get_pressed()[pygame.K_s]:
        from matplotlib import pyplot as plt
        plt.imshow(img.transpose(1, 0, 2), cmap='gray')
        plt.savefig('img.png')

    x, y = pygame.mouse.get_pos()
    if pygame.mouse.get_pressed()[0] and x < 28 * img_upscale:
      x, y = (int(x / img_upscale), int(y / img_upscale))
      img[x, y] = 255
      result: np.ndarray = model(np.expand_dims(img.transpose(1, 0, 2).astype('float32') / 255, axis=0)).numpy()
      predicted = result.argmax()


    screen.fill((125, 125, 125))

    pixel_surface = pygame.surfarray.make_surface(img.repeat(img_upscale, axis=0).repeat(img_upscale, axis=1).repeat(3, axis=2))
    screen.blit(pixel_surface, (0, 0))
    t = font.render(str(predicted), True, (0, 0, 0))
    t_rect = t.get_bounding_rect()
    screen.blit(t, (int(28 * img_upscale + (800 - 28 * img_upscale) / 2 - t_rect.w / 2), int(600 / 2 - t_rect.h / 2)))

    pygame.display.flip()

if __name__ == '__main__':
  main()
