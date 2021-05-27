import sys
import copy
import torch
import torch.nn.parallel
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data
import torch.utils.data.distributed

sys.path.append('.')
from domain_adaptation.adaptation.mdd import MarginDisparityDiscrepancy, ParticlesClassifier
from tools.utils import ForeverDataIterator
from tools.lr_scheduler import StepwiseLR
from domain_adaptation.modules.classifier import Classifier as SourceClassifier

# my utils
from utils.utils import load_pickle, save_pickle
from utils.training_stats import TrainingStats
from utils.validation import validate
from utils.config import Config
from utils.models import Net
from utils.plotting import plot_training_stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'mdd'


def main():
    """
        Before starting this script, the source model must be trained and saved in data/trained_models.
        If Config.train_mdd is set to true, the script trains mdd model and saves it into data/trained_models dir.
        If Config.train_mdd is set to false, the script load mdd model from data/trained_models and validates it
        on the training dataset.

        The training parameters can be set in config.py under the section MDD
    """
    print(f"Training {model_name} classifiers.")

    precision_recall_scores_list = []

    for particle_code in Config.particles_dict:
        particle_name = Config.particles_dict[particle_code]
        training_stats = TrainingStats()
        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        print(f'Creating {model_name} model for {particle_name}')
        n_iters = min(len(datasets_dict['train_source_loader']), len(datasets_dict['train_target_loader']))
        val_loader = datasets_dict['test_target_loader']

        # load source model and set as a backbone
        backbone = Net(Config.n_features)
        source_classifier = SourceClassifier(backbone, Config.n_features, classifier_type='source').to(device)
        source_classifier.load_state_dict(
            torch.load(f'{Config.source_model_fp}_{particle_name}.pt',
                       map_location=torch.device(device)))
        backbone = source_classifier.backbone

        # create model
        classifier = ParticlesClassifier(backbone, Config.n_classes, bottleneck_dim=Config.mdd_bottleneck_dim,
                                         width=Config.mdd_bottleneck_dim).to(device)
        mdd = MarginDisparityDiscrepancy(Config.mdd_margin).to(device)

        # define optimizer and lr_scheduler
        optimizer = Adam(classifier.get_parameters(), Config.mdd_lr, weight_decay=Config.mdd_weight_decay)
        lr_scheduler = StepwiseLR(optimizer, init_lr=Config.mdd_lr, gamma=Config.mdd_lr_gamma, decay_rate=0.75)

        best_test_loss = 100

        # start training
        if Config.train_mdd:
            for epoch in range(Config.mdd_epochs):

                train_source_iter = ForeverDataIterator(datasets_dict['train_source_loader'])
                train_target_iter = ForeverDataIterator(datasets_dict['train_target_loader'])
                test_source_iter = ForeverDataIterator(datasets_dict['test_source_loader'])
                test_target_iter = ForeverDataIterator(datasets_dict['test_target_loader'])

                classifier.train()
                mdd.train()

                criterion = nn.CrossEntropyLoss().to(device)

                loss_item, loss_test_item, classification_loss_item, classification_loss_test_item, trans_loss_item, trans_loss_test_item = 0, 0, 0, 0, 0, 0

                for i in range(n_iters):
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    x_s, labels_s = next(train_source_iter)
                    x_t, labels_t = next(train_target_iter)

                    x_s = x_s.to(device)
                    x_t = x_t.to(device)
                    labels_s = labels_s.to(device)
                    labels_t = labels_t.to(device)

                    # compute output
                    x = torch.cat((x_s, x_t), dim=0)
                    outputs, outputs_adv = classifier(x)
                    y_s, y_t = outputs.chunk(2, dim=0)
                    y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

                    # compute cross entropy loss on source domain
                    cls_loss = criterion(y_s, labels_s)

                    # compute margin disparity discrepancy between domains
                    transfer_loss = mdd(y_s, y_s_adv, y_t, y_t_adv)
                    loss = cls_loss + transfer_loss * Config.mdd_trade_off

                    # testing iterator
                    x_s_test, labels_s_test = next(test_source_iter)
                    x_t_test, labels_t_test = next(test_target_iter)

                    x_s_test = x_s_test.to(device)
                    x_t_test = x_t_test.to(device)
                    labels_s_test = labels_s_test.to(device)
                    labels_t_test = labels_t_test.to(device)

                    # compute output
                    x_test = torch.cat((x_s_test, x_t_test), dim=0)
                    outputs_test, outputs_adv_test = classifier(x_test)
                    y_s_test, y_t_test = outputs_test.chunk(2, dim=0)
                    y_s_adv_test, y_t_adv_test = outputs_adv_test.chunk(2, dim=0)

                    # compute cross entropy loss on source domain
                    cls_loss_test = criterion(y_s_test, labels_s_test)

                    # compute margin disparity discrepancy between domains
                    transfer_loss_test = mdd(y_s_test, y_s_adv_test, y_t_test, y_t_adv_test)
                    loss_test = cls_loss_test + transfer_loss_test * Config.mdd_trade_off

                    # updating tracked losses
                    loss_item += loss.item()
                    loss_test_item += loss_test.item()
                    classification_loss_item += cls_loss.item()
                    classification_loss_test_item += cls_loss_test.item()
                    trans_loss_item += transfer_loss.item()
                    trans_loss_test_item += transfer_loss_test.item()

                    # compute gradient and do step
                    classifier.step()
                    loss.backward()
                    optimizer.step()

                training_stats.update('epoch', epoch)
                training_stats.update('loss', loss_item / n_iters)
                training_stats.update('loss_test', loss_test_item / n_iters)
                training_stats.update('classification_loss', classification_loss_item / n_iters)
                training_stats.update('classification_loss_test', classification_loss_test_item / n_iters)
                training_stats.update('trans_loss', trans_loss_item / n_iters)
                training_stats.update('trans_loss_test', trans_loss_test_item / n_iters)
                training_stats.update('precision_recall_source',
                                      validate(datasets_dict['test_source_loader'], classifier, loud=False))
                training_stats.update('precision_recall_target',
                                      validate(datasets_dict['test_target_loader'], classifier, loud=False))
                training_stats.update('precision_recall_source_train',
                                      validate(datasets_dict['train_source_loader'], classifier, loud=False))
                training_stats.update('precision_recall_target_train',
                                      validate(datasets_dict['train_target_loader'], classifier, loud=False))
                print(f'particle_type: {particle_name}, epoch: {epoch}/{Config.mdd_epochs}')

                if loss_test < best_test_loss:
                    best_test_loss = loss_test
                    best_model = copy.deepcopy(classifier.state_dict())
                    torch.save(best_model, f'{Config.mdd_model_fp}_{particle_name}.pt')


        else:
            classifier.load_state_dict(torch.load(f'{Config.mdd_model_fp}_{particle_name}.pt',
                                                  map_location=torch.device(device)))
            best_model = copy.deepcopy(classifier.state_dict())

        # evaluate on test set
        classifier.load_state_dict(best_model)
        precision_recall_auc_score = validate(val_loader, classifier, print_classification_report=False, loud=False)
        precision_recall_scores_list.append(precision_recall_auc_score)

        print(f'-------------')
        print(f'precision_recall for {particle_name} classifier: {precision_recall_auc_score}\n')
        print(f'-------------')

        training_stats_df = training_stats.get_training_stats_df()
        save_pickle(training_stats_df,
                    f'{Config.source_fp}/pickles/training_stats/training_stats_df_{model_name}_{particle_name}.pkl')
        plot_training_stats(training_stats_df, ['loss', 'loss_test'], model_name=model_name,
                            particle_name=particle_name, label='losses')
        plot_training_stats(training_stats_df, ['precision_recall_source', 'precision_recall_target', 'loss_test'],
                            model_name=model_name, particle_name=particle_name, label='percisions')

    print(
        f'Mean precision_recall for MDD classifier: {sum(precision_recall_scores_list) / len(precision_recall_scores_list)}')


if __name__ == '__main__':
    main()

