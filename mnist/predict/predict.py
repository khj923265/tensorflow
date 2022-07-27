import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:10000]
test_labels = test_labels[:10000]

train_images = train_images[:10000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:10000].reshape(-1, 28 * 28) / 255.0

path = '../model/'
test_model_name = 'test_model.h5'
newModel = tf.keras.models.load_model(path + test_model_name)

predict = newModel.predict(test_images)
print(predict)

exit()
