from keras import backend as K
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, Flatten, InputLayer, Reshape
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, Nadam, RMSprop

class CustomMetrics(Callback):
    def __init__(self, validation_data: tuple, test_generator, diff=None):
        self.test_generator = test_generator
        self.diff = diff
        self.x_test = validation_data[0]
        self.y_test_categorical = validation_data[1]
        
        
    def on_train_begin(self, logs={}):
        self.val_f1s = []
 
    def on_epoch_end(self, epoch, logs={}):        
        # y_true_categorical = self.validation_data[1]
        # y_pred = (np.asarray(self.model.predict_classes(self.x_test[0])))
        y_pred = np.asarray(self.model.predict_generator(self.test_generator, workers=8, use_multiprocessing=1))
        pred_labels = np.array([np.argmax(each) for each in y_pred])
        y_test = np.argmax(self.y_test_categorical, axis=1)[:self.diff]
        _val_f1 = f1_score(y_test, pred_labels, average="macro")
        self.val_f1s.append(_val_f1)
        print(" — val_f1_macro: {}".format(_val_f1))
        
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def train_baseline(epochs, output_dim, input_dim, batch_size, optimizer, loss, train_generator, test_generator, class_weights, custom_metrics):
    model = Sequential()
    model.add(Dense(output_dim, input_shape=(input_dim,), batch_size=batch_size, activation='softmax')) 
    model.compile(
              optimizer=optimizer, # Adagrad(lr=0.1)
              loss=loss,
              metrics=['accuracy'])
    
    history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    verbose=1,
    callbacks=[custom_metrics],
    validation_data=test_generator,
    validation_steps=len(test_generator),
    class_weight=class_weights,
    workers=8,
    use_multiprocessing=True
    )
    return history
