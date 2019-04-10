import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC


def plot_model_history(history, custom_metrics=None):
    # summarize history for accuracy
    plt.plot(history.history['acc']) # acc
    plt.plot(history.history['val_acc']) # val_acc
    if custom_metrics:
        plt.plot(custom_metrics.val_f1s)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', "f1_test"], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def plot_confusion_matrix(y_test, preds, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_test, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_decision_function(X, y, ax, plot_step=0.02):
    svc = LinearSVC().fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
    
def plot_embedding(embeddings, df, cmap="Set1", s=0.3, alpha=1, ax=None, title="2D embeddings"):
    df["emb_x"] = embeddings[:, 0]
    df["emb_y"] = embeddings[:, 1]
    
    for _id, color in zip(df["group_id"].unique(), ['red', 'green', 'blue']):
        x = df[df["group_id"]==_id]["emb_x"]
        y = df[df["group_id"]==_id]["emb_y"]   
        label = df[df["group_id"]==_id]["group_name"].iloc[0]
        ax.scatter(x, y, c=color, s=s, label=label,
                   alpha=alpha, edgecolors='k')
        ax.legend()
    plt.setp(ax, xticks=[], yticks=[])
    plt.title(title, fontsize=18)