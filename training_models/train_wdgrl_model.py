import copy
import torch
from torch import nn
from torch.autograd import grad
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch import abs as t_abs
from tools.utils import ForeverDataIterator
from domain_adaptation.modules.classifier import Classifier as SourceClassifier

# my utils
from utils.utils import load_pickle, save_pickle
from utils.training_stats import TrainingStats
from utils.validation import validate
from utils.config import Config
from utils.models import Net, wdgrl_critic
from utils.plotting import plot_training_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'wdgrl'


def main():
    """
        Before starting this script, the source model must be trained and saved in data/trained_models.
        If Config.train_wdgrl is set to true, the script trains wdgrl model and saves it into data/trained_models dir.
        If Config.train_wdgrl is set to false, the script load wdgrl model from data/trained_models and validates it
        on the training dataset.

        The training parameters can be set in config.py under the section WDGRL
    """
    print(f"Training {model_name} classifiers.")

    precision_scores_list = []

    for particle_code in Config.particles_dict:
        particle_name = Config.particles_dict[particle_code]
        training_stats = TrainingStats()
        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        print(f'Creating {model_name} model for {particle_name}')
        n_iters = min(len(datasets_dict['train_source_loader']), len(datasets_dict['train_target_loader']))
        val_loader = datasets_dict['test_target_loader']

        # load source model and set as a backbone
        backbone = Net(Config.n_features)
        classifier = SourceClassifier(backbone, Config.n_features, classifier_type='source').to(device)
        classifier.load_state_dict(
            torch.load(f'{Config.source_model_fp}_{particle_name}.pt', map_location=torch.device(device)))
        classifier = classifier.backbone

        critic = wdgrl_critic.to(device=device)

        critic_optim = torch.optim.RMSprop(critic.parameters(), lr=0.0001)
        clf_optim = torch.optim.Adam(classifier.parameters(), lr=0.0001)
        clf_criterion = nn.CrossEntropyLoss()

        best_test_loss = 100

        # start training
        if Config.train_wdgrl:
            for epoch in range(Config.wdgrl_epochs):

                train_source_iter = ForeverDataIterator(datasets_dict['train_source_loader'])
                train_target_iter = ForeverDataIterator(datasets_dict['train_target_loader'])
                test_source_iter = ForeverDataIterator(datasets_dict['test_source_loader'])
                test_target_iter = ForeverDataIterator(datasets_dict['test_target_loader'])

                feature_extractor = classifier.feature_extractor
                discriminator = classifier.discriminator

                loss_item, loss_test_item, classification_loss_item, classification_loss_test_item, trans_loss_item, trans_loss_test_item = 0, 0, 0, 0, 0, 0

                for i in range(n_iters):

                    source_x, source_y = next(train_source_iter)
                    target_x, target_y = next(train_target_iter)

                    source_x = source_x.to(device=device)
                    source_y = source_y.to(device=device)
                    target_x = target_x.to(device=device)

                    # Train critic
                    set_requires_grad(classifier, requires_grad=False)
                    set_requires_grad(critic, requires_grad=True)

                    with torch.no_grad():
                        h_s = feature_extractor(source_x).data.view(source_x.shape[0], -1)
                        h_t = feature_extractor(target_x).data.view(target_x.shape[0], -1)
                    h_s = h_s.to(device=device)
                    h_t = h_t.to(device=device)

                    for _ in range(Config.wdgrl_k_critic):
                        gp = gradient_penalty(critic, h_s, h_t)

                        critic_s = critic(h_s)
                        critic_t = critic(h_t)
                        wasserstein_distance = critic_s.mean() - critic_t.mean()

                        critic_cost = -wasserstein_distance + Config.wdgrl_gamma * gp

                        critic_optim.zero_grad()
                        critic_cost.backward()
                        critic_optim.step()

                    # Train classifier
                    set_requires_grad(feature_extractor, requires_grad=True)
                    set_requires_grad(critic, requires_grad=False)

                    for _ in range(Config.wdgrl_k_clf):
                        source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
                        target_features = feature_extractor(target_x).view(target_x.shape[0], -1)

                        source_preds = discriminator(source_features)
                        clf_loss = clf_criterion(source_preds, source_y)
                        wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()

                        loss = clf_loss + Config.wdgrl_wd_clf * t_abs(wasserstein_distance)
                        clf_optim.zero_grad()
                        loss.backward()
                        clf_optim.step()

                    # testing iterator
                    x_s_test, labels_s_test = next(test_source_iter)
                    x_t_test, _ = next(test_target_iter)

                    x_s_test = x_s_test.to(device)
                    x_t_test = x_t_test.to(device)
                    labels_s_test = labels_s_test.to(device)

                    source_features_test = feature_extractor(x_s_test).view(x_s_test.shape[0], -1)
                    target_features_test = feature_extractor(x_t_test).view(x_t_test.shape[0], -1)

                    source_preds_test = discriminator(source_features_test)
                    clf_loss_test = clf_criterion(source_preds_test, labels_s_test)
                    wasserstein_distance_test = critic(source_features_test).mean() - critic(
                        target_features_test).mean()

                    loss_test = clf_loss_test + Config.wdgrl_wd_clf * t_abs(wasserstein_distance_test)

                    # updating tracked losses
                    loss_item += loss.item()
                    loss_test_item += loss_test.item()
                    classification_loss_item += clf_loss.item()
                    classification_loss_test_item += clf_loss_test.item()
                    trans_loss_item += wasserstein_distance.item()
                    trans_loss_test_item += wasserstein_distance_test.item()

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
                print(f'particle_type: {particle_name}, epoch: {epoch}/{Config.wdgrl_epochs}')

                if loss_test < best_test_loss:
                    best_test_loss = loss_test
                    best_model = copy.deepcopy(classifier.state_dict())
                    torch.save(best_model, f'{Config.wdgrl_model_fp}_{particle_name}.pt')

        else:
            classifier.load_state_dict(torch.load(f'{Config.wdgrl_model_fp}_{particle_name}.pt',
                                                  map_location=torch.device(device)))
            best_model = copy.deepcopy(classifier.state_dict())

        # evaluate on test set
        classifier.load_state_dict(best_model)
        precision_auc_score = validate(val_loader, classifier, print_classification_report=False, loud=False)
        precision_scores_list.append(precision_auc_score)
        print(f'-------------')
        print(f'precision for {particle_name} classifier: {precision_auc_score}\n')
        print(f'-------------')
        training_stats_df = training_stats.get_training_stats_df()
        save_pickle(training_stats_df,
                    f'{Config.source_fp}/pickles/training_stats/training_stats_df_{model_name}_{particle_name}.pkl')
        plot_training_stats(training_stats_df, ['loss', 'loss_test'], model_name=model_name,
                            particle_name=particle_name, label='losses')
        plot_training_stats(training_stats_df, ['precision_recall_source', 'precision_recall_target', 'loss_test'],
                            model_name=model_name, particle_name=particle_name, label='percisions')

    print(f'Mean precision for WDGRL classifier: {sum(precision_scores_list) / len(precision_scores_list)}')


def gradient_penalty(critic, h_s, h_t):
    alpha = torch.rand(h_s.size(0), 1).to(device=device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    predictions = critic(interpolates)
    gradients = grad(predictions, interpolates,
                     grad_outputs=torch.ones_like(predictions),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1)**2).mean()
    return penalty


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


if __name__ == '__main__':
    main()
