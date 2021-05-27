import time
import sys
import copy
import torch
import torch.nn.parallel
from torch.optim import Adam
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from domain_adaptation.modules.classifier import Classifier
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.lr_scheduler import StepwiseLR
from utils.data_loader import prepare_datasets_dict
from utils.config import Config
from utils.models import Net
from utils.validation import validate
from utils.utils import load_pickle

sys.path.append('')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Creates a pickle with the dictionary of datasets which is then used in another scripts.
    Creates default dense deep neural network which doesn't implements domain adaptation
    techniques and saves it into pickle.
    """

    precision_scores_list = []
    print("Training source classifiers.")
    for particle_code in Config.particles_dict:
        particle_name = Config.particles_dict[particle_code]
        print(f'Creating source model for {particle_name}')

        # create dataset dict
        prepare_datasets_dict(Config.test_size, Config.seed, Config.batch_size, class_to_predict=particle_code)

        # load dataset dicts
        datasets_dict = load_pickle(
            f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        training_weights = load_pickle(
            f'{Config.training_data_fp}/training_weights_{particle_name}.pkl')

        val_loader = datasets_dict['test_source_loader']
        train_source_iter = ForeverDataIterator(datasets_dict['train_source_loader'])
        n_iters = len(datasets_dict['train_source_loader'])

        # create model
        backbone = Net(Config.n_features)
        classifier = Classifier(backbone, Config.n_features, classifier_type='source').to(device)

        # define optimizer and lr scheduler
        optimizer = Adam(classifier.get_parameters(), Config.source_lr)
        lr_sheduler = StepwiseLR(optimizer, init_lr=Config.source_lr, gamma=0.0003, decay_rate=0.75)

        # start training
        if Config.train_source:
            for epoch in range(Config.source_epochs):

                # train for one epoch
                train(train_source_iter, classifier, optimizer,
                      lr_sheduler, epoch, training_weights, n_iters)

            model = copy.deepcopy(classifier.state_dict())
            torch.save(model, f'{Config.source_model_fp}_{particle_name}.pt')

        else:
            classifier.load_state_dict(torch.load(f'{Config.source_model_fp}_{particle_name}.pt',
                                                  map_location=torch.device(device)))
            model = copy.deepcopy(classifier.state_dict())

        # evaluate on test set
        classifier.load_state_dict(model)
        precision_auc_score = validate(val_loader, classifier, print_classification_report=True, loud=False)
        precision_scores_list.append(precision_auc_score)
        print(f'precision for {particle_name} classifier: {precision_auc_score}')
    print(f'Mean precision: {sum(precision_scores_list) / len(precision_scores_list)}')


def train(train_source_iter: ForeverDataIterator, model: Classifier, optimizer: Adam,
          lr_sheduler: StepwiseLR, epoch: int, training_weights, n_iters: int) -> None:
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        n_iters,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(n_iters):
        if lr_sheduler is not None:
            lr_sheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        y_s, f_s = model(x_s)
        y_s = y_s.to(device)

        cls_loss = F.cross_entropy(y_s, labels_s, weight=torch.FloatTensor(training_weights).to(device))
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % Config.source_print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()

