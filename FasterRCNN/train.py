from keras.layers import Input
from keras.optimizers import Adam

from FasterRCNN.model import shared_cnn_model, rpn_model, cls_model
from FasterRCNN.losses import rpn_cls_loss, rpn_regr_loss, cls_regr_loss


img_input = Input((None, None, None))
roi_input = Input((4,))  # the output of rpn

shared_model = shared_cnn_model()
rpn = rpn_model(img_input, shared_model)
classifier = cls_model(img_input, roi_input, shared_model)

rpn.compile(loss=[rpn_cls_loss, rpn_regr_loss], optimizer=Adam(lr=1e-5, decay=5e-4))
classifier.compile(loss=['categorical_crossentropy', cls_regr_loss], optimizer=Adam(lr=1e-5, decay=5e-4))

