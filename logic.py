import numpy as np
import  tensorflow as tf


def get_center(vector, fv_model):
    return np.mean(fv_model.predict(tf.constant(vector)))


def get_label(input, nbr_model, fv_model):
    c = get_center(input, fv_model)
    i = nbr_model.predict(np.array(c).reshape(-1, 1))
    label = "human" if i == 0 else ("pet" if i == 1 else "elephant")
    print(label)
    return label