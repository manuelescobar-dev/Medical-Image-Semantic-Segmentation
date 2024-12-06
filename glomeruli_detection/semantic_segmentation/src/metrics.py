import numpy as np
import matplotlib.pyplot as plt
import itertools

def confusion_matrix(predicted, labels):
    """
    Compute the confusion matrix for binary classification.

    Parameters:
    - predicted: Array-like of predicted labels (0 or 1).
    - labels: Array-like of true labels (0 or 1).

    Returns:
    - A NumPy array with counts of TP, TN, FP, and FN.
    """
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()
    return np.array([tp, tn, fp, fn], dtype=np.uint32)

def recall(conf_matrix):
    """
    Calculate the recall metric.

    Parameters:
    - conf_matrix: Confusion matrix as a NumPy array.

    Returns:
    - Recall score.
    """
    tp = conf_matrix[0]
    fn = conf_matrix[3]
    return tp / (tp + fn)

def precision(conf_matrix):
    """
    Calculate the precision metric.

    Parameters:
    - conf_matrix: Confusion matrix as a NumPy array.

    Returns:
    - Precision score.
    """
    tp = conf_matrix[0]
    fp = conf_matrix[2]
    return tp / (tp + fp)

def accuracy(conf_matrix):
    """
    Calculate the accuracy metric.

    Parameters:
    - conf_matrix: Confusion matrix as a NumPy array.

    Returns:
    - Accuracy score.
    """
    tp = conf_matrix[0]
    tn = conf_matrix[1]
    fp = conf_matrix[2]
    fn = conf_matrix[3]
    return (tp + tn) / (tp + fn + fp + tn)

def f1(conf_matrix):
    """
    Calculate the F1 score.

    Parameters:
    - conf_matrix: Confusion matrix as a NumPy array.

    Returns:
    - F1 score.
    """
    prec = precision(conf_matrix)
    rec = recall(conf_matrix)
    return (2 * prec * rec) / (prec + rec)

def iou(conf_matrix):
    """
    Calculate the Intersection over Union (IoU) metric.

    Parameters:
    - conf_matrix: Confusion matrix as a NumPy array.

    Returns:
    - IoU score.
    """
    tp = conf_matrix[0]
    fp = conf_matrix[2]
    fn = conf_matrix[3]
    return tp / (tp + fn + fp)

def evaluate(conf_matrix):
    """
    Print a table with various evaluation metrics based on the confusion matrix.

    Parameters:
    - conf_matrix: Confusion matrix as a NumPy array.
    """
    tp = conf_matrix[0]
    tn = conf_matrix[1]
    fp = conf_matrix[2]
    fn = conf_matrix[3]

    # Print confusion matrix
    print("Confusion Matrix")
    print("                 Predicted")
    print("                 1        0")
    print(f"Actual 1: {tp:>10} {fn:>10}")
    print(f"       0: {fp:>10} {tn:>10}")

    # Print metrics
    acc = accuracy(conf_matrix)
    rec = recall(conf_matrix)
    prec = precision(conf_matrix)
    f1_score = f1(conf_matrix)
    iou_score = iou(conf_matrix)

    print(f"\nAccuracy: {acc:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"IoU: {iou_score:.2f}")

def plot_confusion_matrix(cm, class_names=['glomeruli', 'non-glomeruli'], title='Confusion Matrix', normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters:
    - cm: Confusion matrix (2D array or list of lists).
    - class_names: List of class names.
    - title: Title of the plot.
    - normalize: Whether to normalize the confusion matrix.
    - cmap: Color map to use for the plot.
    """
    
    if normalize:
        cm = np.array(cm).astype('float') / np.sum(cm, axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # In case of division by zero

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    # Example usage:
    cm = [[647182, 56660], [73830, 5539310]]

    plot_confusion_matrix(cm, normalize=True)