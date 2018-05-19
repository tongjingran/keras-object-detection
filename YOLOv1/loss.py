import tensorflow as tf
import keras.backend as K

# y_true == (bs,7,7,30)
def custom_loss(y_true, y_pred):
    lambda_coord = 5
    lambda_noobj = 0.5
    