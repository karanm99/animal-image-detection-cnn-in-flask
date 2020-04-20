import numpy as np
from keras.preprocessing import image


def predict_img(img_path):
    import tensorflow as tf
    cnn = tf.keras.models.load_model('cnn_model.h5')

    # summarize model
    # print(cnn.summary())

    test_image = image.load_img(img_path, target_size=(64, 64, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    # training_set.class_indices
    if result[0][0] == 1:
        prediction = 'Dog..!!'
    else:
        prediction = 'Cat..!!'
    return prediction

