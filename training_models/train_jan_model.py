import sys
import copy
import torch
import torch.nn.parallel
from torch.optim import Adam
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

sys.path.append('.')
from domain_adaptation.adaptation.jan import JointMultipleKernelMaximumMeanDiscrepancy, ParticlesClassifier, Theta
from domain_adaptation.modules.kernels import GaussianKernel
from domain_adaptation.modules.classifier import Classifier as SourceClassifier
from tools.utils import ForeverDataIterator
from tools.lr_scheduler import StepwiseLR

# my utils
from utils.utils import load_pickle, save_pickle
from utils.training_stats import TrainingStats
from utils.validation import validate
from utils.config import Config
from utils.models import Net
from utils.plotting import plot_training_stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'jan'


def main():
    """
        Before starting this script, the source model must be trained and saved in data/trained_models.
        If Config.train_jan is set to true, the script trains dan model and saves it into data/trained_models dir.
        If Config.train_jan is set to false, the script load dan model from data/trained_models and validates it
        on the training dataset.

        The training parameters can be set in config.py under the section JAN
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
            torch.load(f'{Config.source_model_fp}_{Config.particles_dict[particle_code]}.pt',
                       map_location=torch.device(device)))
        backbone = source_classifier.backbone

        num_classes = Config.n_classes
        classifier = ParticlesClassifier(backbone, num_classes).to(device)

        # define loss function
        if Config.jan_adversarial:
            thetas = [Theta(dim).to(device) for dim in (classifier.features_dim, num_classes)]
        else:
            thetas = None
        jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=Config.jan_linear, thetas=thetas
        ).to(device)

        parameters = classifier.get_parameters()
        if thetas is not None:
            parameters += [{"params": theta.parameters(), 'lr_mult': 0.1} for theta in thetas]

        # define optimizer
        optimizer = Adam(parameters, Config.jan_lr, weight_decay=Config.jan_weight_decay)
        lr_sheduler = StepwiseLR(optimizer, init_lr=Config.jan_lr, gamma=0.0003, decay_rate=0.75)

        best_test_loss = 100

        # start training
        if Config.train_jan:
            for epoch in range(Config.jan_epochs):

                # train for one epoch
                train_source_iter = ForeverDataIterator(datasets_dict['train_source_loader'])
                train_target_iter = ForeverDataIterator(datasets_dict['train_target_loader'])
                test_source_iter = ForeverDataIterator(datasets_dict['test_source_loader'])
                test_target_iter = ForeverDataIterator(datasets_dict['test_target_loader'])

                # switch to train mode
                classifier.train()
                jmmd_loss.train()

                loss_item, loss_test_item, classification_loss_item, classification_loss_test_item, trans_loss_item, trans_loss_test_item = 0, 0, 0, 0, 0, 0

                for i in range(n_iters):
                    lr_sheduler.step()

                    x_s, labels_s = next(train_source_iter)
                    x_t, _ = next(train_target_iter)

                    x_s = x_s.to(device)
                    x_t = x_t.to(device)
                    labels_s = labels_s.to(device)

                    # compute output
                    x = torch.cat((x_s, x_t), dim=0)
                    y, f = classifier(x)
                    y_s, y_t = y.chunk(2, dim=0)
                    f_s, f_t = f.chunk(2, dim=0)

                    cls_loss = F.cross_entropy(y_s, labels_s)
                    transfer_loss = jmmd_loss(
                        (f_s, F.softmax(y_s, dim=1)),
                        (f_t, F.softmax(y_t, dim=1))
                    )
                    loss = cls_loss + transfer_loss * Config.jan_trade_off

                    # testing iterator
                    x_s_test, labels_s_test = next(test_source_iter)
                    x_t_test, _ = next(test_target_iter)

                    x_s_test = x_s_test.to(device)
                    x_t_test = x_t_test.to(device)
                    labels_s_test = labels_s_test.to(device)

                    x_test = torch.cat((x_s_test, x_t_test), dim=0)
                    y_test, f_test = classifier(x_test)
                    y_s_test, y_t_test = y_test.chunk(2, dim=0)
                    f_s_test, f_t_test = f_test.chunk(2, dim=0)

                    cls_loss_test = F.cross_entropy(y_s_test, labels_s_test)
                    transfer_loss_test = jmmd_loss(
                        (f_s_test, F.softmax(y_s_test, dim=1)),
                        (f_t_test, F.softmax(y_t_test, dim=1))
                    )
                    loss_test = cls_loss_test + transfer_loss_test * Config.jan_trade_off

                    # updating tracked losses
                    loss_item += loss.item()
                    loss_test_item += loss_test.item()
                    classification_loss_item += cls_loss.item()
                    classification_loss_test_item += cls_loss_test.item()
                    trans_loss_item += transfer_loss.item()
                    trans_loss_test_item += transfer_loss_test.item()

                    # compute gradient and do step
                    optimizer.zero_grad()
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
                print(f'particle_type: {particle_name}, epoch: {epoch}/{Config.jan_epochs}')

                if loss_test < best_test_loss:
                    best_test_loss = loss_test
                    best_model = copy.deepcopy(classifier.state_dict())
                    torch.save(best_model, f'{Config.jan_model_fp}_{particle_name}.pt')

                training_stats_df = training_stats.get_training_stats_df()
                plot_training_stats(training_stats_df, ['loss', 'loss_test'], model_name=model_name,
                                    particle_name=particle_name, label='losses')
                plot_training_stats(training_stats_df,
                                    ['precision_recall_source', 'precision_recall_target', 'loss_test'],
                                    model_name=model_name, particle_name=particle_name, label='percisions')

        else:
            classifier.load_state_dict(torch.load(f'{Config.jan_model_fp}_{particle_name}.pt',
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
        f'Mean precision_recall for JAN classifier: {sum(precision_recall_scores_list) / len(precision_recall_scores_list)}')


if __name__ == '__main__':
    main()

