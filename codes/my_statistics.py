from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sn
import matplotlib.pyplot as plt
from inspect import signature
import uuid


class My_Statistics(object):
    """docstring forMy_Statistics."""

    def __init__(self, arg=None):
        self.arg = arg
        self.graph_save_path = "../graphs/"
        self.show_plt = False

    def print_all_statistics(self, y_true, y_pred, target_names, figsize=(10, 10), title="Confusion matrix", cmap=plt.cm.Reds):
        """
        Overviews:
            This plots the statistics such as confusion matrix, mean accuracy, classification report (class wise precision, recall, f1-score and support)
        Params:
            y_true: true class labels. e.g. the shape could be (204,). So this is a row array
            y_pred: predicted class labels. e.g. the shape could be (204,). So this is a row array
            target_names: list of class names. e.g target_names = ["0", "1", "2"]
        """
        print("plotting statistics ... ...")
        self.print_classification_report(y_true, y_pred, target_names)
        self.plot_confusion_matrix(
            y_true, y_pred, target_names, figsize, title, cmap)
        self.plot_precision_recall_curve(y_true, y_pred)

    def print_classification_report(self, y_true, y_pred, target_names):
        print("plotting classification report ... ...")
        print(classification_report(y_true=y_true,
                                    y_pred=y_pred, target_names=target_names))

    def print_mean_accuracy(self, classifier, x_df, y_array):
        print("printing means accuracy of the classifier ... ...")
        accuracy = classifier.score(x_df, y_array)
        print("mean accuracy: ", accuracy)

    def plot_precision_recall_curve(self, y_true, y_score, figsize=(10, 10), save_path=None):
        """
        Params:
            y_true: a row vector of true labels. e.g. (204, )
            y_score: a row vector of predicted labels. e.g. (204, )
            classes: a list of numeric class labels. e.g. [0, 1, 2]
        Preconditions:
            hard coded classes used [0, 1, 2]
            y_true and y_score must be same condition.
        """
        print("Plotting the micro-averaged Precision-Recall curve ... ...")
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        classes = [0, 1, 2]
        # from (204, ) to (204, 3)
        y_true = label_binarize(y_true.transpose(), classes=classes)
        # from (204, ) to (204, 3)
        y_score = label_binarize(y_score.transpose(), classes=classes)
        n_classes = y_true.shape[1]
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(
                y_true[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
                                                                        y_score.ravel())
        average_precision["micro"] = average_precision_score(y_true, y_score,
                                                             average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(
            average_precision["micro"]))

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.figure(figsize=figsize)
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                         **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
        plt.savefig(self.graph_save_path + str(uuid.uuid4()) + ".png")
        if(self.show_plt):
            plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, target_names, figsize=(10, 10), title="Confusion matrix", cmap=plt.cm.Reds):
        print("plotting confusion matrix ... ...")
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm)
        plt.figure(figsize=figsize)
        sn.set(font_scale=1.4)
        ax = sn.heatmap(df_cm, annot=True, fmt="g", cmap=cmap)
        ax.set_ylim(len(target_names), 0)
        ax.set(xlabel='Predicted label', ylabel='True label', title=title)
        plt.savefig(self.graph_save_path + str(uuid.uuid4()) + ".png")
        if(self.show_plt):
            plt.show()
