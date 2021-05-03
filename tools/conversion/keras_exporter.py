from keras.layers.advanced_activations import ReLU
from keras.models import Model
import tensorflow as tf


def export_keras_to_tflite(keras_file, tflite_file):
    model = tf.keras.models.load_model(str("{}".format(keras_file)))

    # hack to make an add 0 to the model in a way that tflite does not remove it.
    # Tflite will change the add 1
    # subtract 1 to an add 0. Normally, add 0 will be deleted by tflite converter.
    # The reason we need an add 0 is
    # because GPU delegate gives wrong output without the add 0
    #
    # ** ReLU may not be needed, depending on TF version.

    # rename output layers to order them in alphabetical order
    outputlen = len(model.outputs)
    outputs_names = "abcdefghijklmnopqrstuvw"
    outputs_names = ["aa" + x for x in outputs_names]
    for i in range(0, outputlen):
        relu = ReLU(name="relu" + str(i), negative_slope=0.999)(model.outputs[i])
        adder = tf.keras.layers.Lambda(lambda x: x + tf.constant(1.0))(relu)
        subtracter = tf.keras.layers.Lambda(
            lambda x: x + tf.constant(-1.0), name=outputs_names[i]
        )(adder)
        model.outputs.append(subtracter)
    m = Model(model.inputs, model.outputs)
    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)
