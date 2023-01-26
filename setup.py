from tensorflow import keras
import sklearn
import pickle


def get_fv_model():
    return keras.models.load_model(
        "./weights/fv_model.keras",
        compile=False
    )


def get_nbr_model():
    loaded_model = pickle.load(open('./weights/nbr_model.keras', 'rb'))
    return loaded_model