import keras.layers
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:10000]
test_labels = test_labels[:10000]

train_images = train_images[:10000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:10000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


model = create_model()

model.fit(test_images, test_labels, epochs=10)

path = '../model/'
test_model_name = 'test_model.h5'
model.save(path + test_model_name)

exit()

# 가중치만 따로 저장시
# def trainTest():
#     path = './weights/train/'
#     model.load_weights(path)
#     loss, acc = model.evaluate(train_images, train_labels, verbose=2)
#     print("복원된 모델의 정확도:{:5.2f}%".format(100 * acc))
#     exit()
#     model.fit(train_images, train_labels, epochs=10)
#     model.save_weights(path)
#
#
# def test():
#     path = './weights/test/'
#     model.load_weights(path)
#     loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#     print("복원된 모델의 정확도:{:5.2f}%".format(100 * acc))
#     exit()
#     model.fit(test_images, test_labels, epochs=10)
#     model.save_weights(path)
