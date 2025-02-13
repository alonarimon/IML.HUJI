import numpy as np

POSITIVE = 1

def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    n_samples = y_true.shape[0]
    return (1/n_samples) * np.sum(np.power(y_pred - y_true, 2))


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise Exception("predicted responses must be in the same size as true")

    miss_num = np.sum(y_true != y_pred)
    if normalize:
        return float(miss_num/y_true.shape[0])
    return float(miss_num)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise Exception("predicted responses must be in the same size as true")

    samples_num = y_true.shape[0]

    # TP_TN = TP + TN = all the agreements between y_true and y_pred
    TP_TN = np.sum(y_true == y_pred)

    accuracy = float(TP_TN / samples_num)
    return accuracy


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
