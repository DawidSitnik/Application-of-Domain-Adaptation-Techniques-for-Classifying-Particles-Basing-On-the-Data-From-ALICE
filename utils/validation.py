import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(val_loader: DataLoader, model, loud=False, print_classification_report=False, return_predictions=False,
             return_probabilities=False):
    """
    Makes validation of the model using validation dataset.

    Parameters:
        - val_loader - DataLoader with validation dataset which consists labels
        - model - model which is going to be validated
        - loud - if set to True prints the score for the model
        - print_classification_report - if set to true prints classification report
        - return_predictions - if set to true returns precision_recall_auc, outputs, targets (float, list, list)
                               else returns precision_recall_auc
        - return_probabilities - if return predictions == True and return_probabilities == True,
                                 list of outputs and targets consists of class probabilities instead of class labels

    Returns:
        - precision recall auc for the model or precision recall auc for the model + lists of outputs and targets
    """

    outputs, outputs_probabilities, targets = [], [], []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (features, target) in enumerate(val_loader):

            features = features.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(features)

            # appending values for classification report
            outputs_probabilities = outputs_probabilities + list(output[:, 1].cpu().detach().numpy())
            _, output = torch.max(output.data, 1)
            outputs = outputs + list(output.cpu().detach().numpy())
            targets = targets + list(target.cpu().detach().numpy())

        precision_recall_auc = average_precision_score(targets, outputs_probabilities)

        if loud:
            print(f'Precision-Recall Score: {precision_recall_auc}')
        if print_classification_report:
            print(f'Classification Report: \n{classification_report(targets, outputs)}')
            print(f'Confusion Matrix: \n{np.round(confusion_matrix(targets, outputs), 2)}')
            print(f'Precision-Recall Score: \n{precision_recall_auc}')

    if return_predictions:
        if not return_probabilities:
            return precision_recall_auc, outputs, targets
        else:
            return precision_recall_auc, outputs_probabilities, targets
    else:
        return precision_recall_auc
