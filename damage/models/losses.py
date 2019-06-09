from tensorflow.keras import backend as K


def recall(y_true, y_pred):
    true_positives_computed = true_positives(y_true, y_pred)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives_computed / (possible_positives + K.epsilon())

def positives(y_true, y_pred):
    return K.sum(y_true)


def negatives(y_true, y_pred):
    return K.mean(1 - y_true)


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def true_negatives(y_true, y_pred):
    y_pred_neg = 1 - K.round(y_pred)
    y_neg = 1 - y_true
    return K.sum(y_neg * y_pred_neg)


def false_negatives(y_true, y_pred):
    y_pred_neg = 1 - K.round(y_pred)
    return K.sum(y_true * y_pred_neg)


def false_positives(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fp = K.sum(y_neg * y_pred_pos)
    return fp
  

def precision(y_true, y_pred):
    true_positives_computed = true_positives(y_true, y_pred)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives_computed / (predicted_positives + K.epsilon())
    return precision


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
