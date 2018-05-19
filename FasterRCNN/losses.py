from keras import backend as K


def rpn_cls_loss(y_true, y_pred):
    is_box_valid = y_true[:, :, :, :9]
    rpn_overlap = y_true[:, :, :, 9:]

    # bce for all anchors
    bce = K.binary_crossentropy(y_pred, rpn_overlap)

    # loss for only valid anchors. is_box_valid is a filter
    cross_ent = is_box_valid * bce

    # number of valid anchors
    n_cls = K.sum(is_box_valid)

    return K.sum(cross_ent) / n_cls


def rpn_regr_loss(y_true, y_pred):
    rpn_regr = y_true[:, :, :, 4 * 9:]
    rpn_overlap = y_true[:, :, :, :4 * 9]

    rpn_regr_lambda = 10

    # smooth l1 loss
    x = rpn_regr - y_pred
    x_bool = K.cast(K.less_equal(K.abs(x), 1.0), K.floatx())
    smooth_l1 = x_bool * (0.5 * x * x) + (1 - x_bool) * (K.abs(x) - 0.5)

    # number of anchors
    n_reg = y_true

    return rpn_regr_lambda * K.sum(rpn_overlap * smooth_l1) / n_reg


def cls_regr_loss(y_true, y_pred):
    regr = y_true[:, 4 * 21:]
    overlap = y_true[:, :4 * 21]

    my_lambda = 5

    # smooth l1 loss
    x = regr - y_pred
    x_bool = K.cast(K.less_equal(K.abs(x), 1.0), K.floatx())
    smooth_l1 = x_bool * (0.5 * x * x) + (1 - x_bool) * (K.abs(x) - 0.5)

    # count activated regr
    n_reg = K.sum(1e-6 + overlap)

    return my_lambda * K.sum(overlap * smooth_l1) / n_reg
