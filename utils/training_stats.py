import pandas as pd


class TrainingStats:
    def __init__(self):
        self.epoch = []
        self.loss = []
        self.loss_test = []
        self.trans_loss = []
        self.trans_loss_test = []
        self.precision_recall_source = []
        self.precision_recall_target = []
        self.classification_loss = []
        self.classification_loss_test = []
        self.precision_recall_target_train = []
        self.precision_recall_source_train = []

    def update(self, metric, value):
        if metric == 'epoch':
            self.epoch.append(value)
        if metric == 'loss':
            self.loss.append(value)
        if metric == 'loss_test':
            self.loss_test.append(value)
        if metric == 'classification_loss':
            self.classification_loss.append(value)
        if metric == 'classification_loss_test':
            self.classification_loss_test.append(value)
        if metric == 'trans_loss':
            self.trans_loss.append(value)
        if metric == 'trans_loss_test':
            self.trans_loss_test.append(value)
        if metric == 'precision_recall_source':
            self.precision_recall_source.append(value)
        if metric == 'precision_recall_target':
            self.precision_recall_target.append(value)
        if metric == 'precision_recall_target_train':
            self.precision_recall_target_train.append(value)
        if metric == 'precision_recall_source_train':
            self.precision_recall_source_train.append(value)

    def get_training_stats_df(self):
        training_stats_df = pd.DataFrame(
            {'epoch': self.epoch,
             'loss': self.loss,
             'loss_test': self.loss_test,
             'trans_loss': self.trans_loss,
             'trans_loss_test': self.trans_loss_test,
             'classification_loss': self.classification_loss,
             'classification_loss_test': self.classification_loss_test,
             'precision_recall_source': self.precision_recall_source,
             'precision_recall_target': self.precision_recall_target,
             'precision_recall_target_train': self.precision_recall_target_train,
             'precision_recall_source_train': self.precision_recall_source_train
             })

        return training_stats_df
