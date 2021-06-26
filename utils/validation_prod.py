import matplotlib.pyplot as plt
from matplotlib.pyplot import axes
from scipy import integrate
import torch
import numpy as np
from utils.config import Config

device = torch.device("cpu")


def validate(val_loader, model):
    outputs = []
    model.eval()
    with torch.no_grad():
        for i, (features, _) in enumerate(val_loader):
            features = features.to(device)
            try:
                output, _ = model(features)
                _, output = torch.max(output.data, 1)
            except:
                output = model(features)
                _, output = torch.max(output.data, 1)

            outputs = outputs + list(output.cpu().detach().numpy())
    return outputs


def gaussian(x, height, center, width):
    return height * np.exp(-(x - center) ** 2 / (2 * width ** 2))


def one_gaussians(x, h1, c1, w1):
    return gaussian(x, h1, c1, w1)


def estimate_classification_quality(data_frame, pdg_code, model_name, particle_name):
    positive_classified = data_frame.query('pdg_code == 1')
    negative_classified = data_frame.query('pdg_code == 0')
    n_ranges = 4
    p_ranges = [0.48272076, 0.58704057, 0.69136038, 0.79568019, 0.9]

    df_array_pos = []
    df_array_neg = []
    df_all = []
    counts_0 = []
    counts_1 = []
    bins = []
    p = []
    p_res = []

    true_positives = []
    false_negative = []
    not_included = []
    false_positives = []
    spec = []

    avg_gauss_prob, avg_model_prob, avg_efficiency, avg_accuracy, avg_f1 = [], [], [], [], []

    optim = np.array([[9.28490710e-02, 4.92932568e+01, 4.5009191e+00, 2.006539930e-03,
                       8.4027208e+01, 5.6641913e+00, 1.25670632e-03, 8.45701248e+01,
                       4.81806393e+00, 2.119629896e-04, 1.89537870e+02, 1.29678890e+01],
                      [9.65255222e-02, 5.03579685e+01, 3.9564949e+00, 3.31816877e-03,
                       7.16847763e+01, 5.60187743e+00, 1.27833193e-03, 8.41476779e+01,
                       5.32993537e+00, 1.21022554e-03, 1.59861195e+02, 1.61194061e+01],
                      [8.53949927e-02, 5.0438144e+01, 4.29894069e+00, 4.76371132e-03,
                       6.36489799e+01, 4.96055621e+00, 1.07104088e-03, 8.38109634e+01,
                       5.10032287e+00, 3.1240542e-03, 1.26808938e+02, 1.00731521e+01],
                      [8.12346697e-02, 5.11918945e+01, 4.52243773e+00, 7.08224843e-03,
                       5.95458302e+01, 4.56349196e+00, 1.01306200e-03, 8.40796827e+01,
                       5.35597779e+00, 4.25516412e-03, 1.03328981e+02, 9.37278245e+00]])
    ranges_lengths = []
    for i in range(n_ranges):
        spec.append(optim[i][3 * pdg_code: 3 * pdg_code + 3])
        df_array_pos.append(positive_classified[
                                positive_classified['P'].between(p_ranges[i], p_ranges[i + 1], inclusive=False)])
        df_array_neg.append(negative_classified[
                                negative_classified['P'].between(p_ranges[i], p_ranges[i + 1], inclusive=False)])

        df_all.append(data_frame[data_frame['P'].between(p_ranges[i], p_ranges[i + 1], inclusive=False)])
        ranges_lengths.append(len(df_array_pos) + len(df_array_neg))

        fig = plt.figure(figsize=(15, 8), dpi=80)
        ax = axes()

        bins_tmp = np.linspace(25, 250, 100)
        counts_tmp, bins_tmp, _ = plt.hist([df_array_neg[i].tpc_signal, df_array_pos[i].tpc_signal], bins=bins_tmp,
                                           alpha=0.9, edgecolor='black', density=True, stacked=True)
        plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.7)
        plt.xlim((0, 250))
        plt.ylim((9e-5, 1))
        plt.xlabel('tpc_signal')
        plt.yscale('log')
        ax.set_facecolor('w')
        plt.title('tpc_signal distribution, P in range (' + str(round(p_ranges[i], 2)) + ', ' + str(
            round(p_ranges[i + 1], 2)) + ']')

        del fig, ax
        # plt.pause(0.1)
        # plt.clf()
        plt.savefig((f'{Config.source_fp}/plots/p_vs_tpc_gauss/{model_name}_{particle_name}_gauss_1.png'))
        counts_0.append(counts_tmp[0])
        counts_1.append(counts_tmp[1])

        new_bin = []
        for j in range(len(bins_tmp) - 1):
            new_bin.append(bins_tmp[j] + (bins_tmp[j + 1] - bins_tmp[j]) / 2)

        bins.append(new_bin)

        p_tmp = integrate.quad(lambda x: one_gaussians(x, *optim[i][3 * pdg_code:3 * pdg_code + 3]), 0, 300)

        p.append(p_tmp[0])
        p_res.append(len(df_array_pos[i]) / (len(df_array_neg[i]) + len(df_array_pos[i])))

        current_data_frame = df_array_pos[i]
        current_data_frame_not_2212 = df_array_neg[i]
        true_positives.append(current_data_frame[
                                  current_data_frame['tpc_signal'].between(spec[i][1] - 3 * spec[i][2],
                                                                           spec[i][1] + 3 * spec[i][2],
                                                                           inclusive=True)])
        false_negative.append(current_data_frame_not_2212[
                                  current_data_frame_not_2212['tpc_signal'].between(
                                      spec[i][1] - 3 * spec[i][2], spec[i][1] + 3 * spec[i][2], inclusive=True)])

        not_included.append(current_data_frame_not_2212[
                                ~current_data_frame_not_2212['tpc_signal'].between(
                                    spec[i][1] - 3 * spec[i][2], spec[i][1] + 3 * spec[i][2], inclusive=True)])

        false_positives.append(current_data_frame[
                                   ~current_data_frame['tpc_signal'].between(
                                       spec[i][1] - 3 * spec[i][2], spec[i][1] + 3 * spec[i][2], inclusive=True)])

        gauss_prob = p[i]
        model_prob = p_res[i]
        efficiency = p_res[i] / p[i]

        # not_included -> true negatives
        tp = len(true_positives[i])
        tn = len(not_included[i])
        fp = len(df_array_pos[i]) - tp
        fn = len(df_array_neg[i]) - tn
        f1 = tp / (tp + 1 / 2 * (fp + fn))

        accuracy = (len(true_positives[i]) + len(not_included[i])) / (len(df_array_pos[i]) + len(df_array_neg[i]))

        avg_gauss_prob.append(gauss_prob)
        avg_model_prob.append(model_prob)
        avg_efficiency.append(efficiency)
        avg_accuracy.append(accuracy)
        avg_f1.append(f1)

        # print("range: " + str(i))
        # print("Gauss center: " + str(spec[i][1]) + " sigma: " + str(optim[i][pdg_code + 3]))
        # print("True positive: " + str(len(true_positives[i])))
        # print("False positives: " + str(len(false_positives[i])))
        # print("False negatives: " + str(len(false_negative[i])))
        # print("True negatives: " + str(len(not_included[i])))
        # print("Gauss probability : " + str(gauss_prob) + " Model probability: " + str(model_prob))
        # print("Efficiency : " + str(efficiency))
        # print("Accuracy : " + str(accuracy))

        fig = plt.figure(figsize=(15, 8), dpi=80)
        ax = axes()
        plt.hist(bins[i], bins[i], weights=(counts_1[i] - counts_0[i]),
                 alpha=0.9, edgecolor='black')
        plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.7)
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.7)
        plt.xlim((0, 250))
        plt.ylim((9e-5, 1))
        plt.xlabel('tpc_signal')

        plt.title('tpc_signal distribution, P in range (' + str(round(p_ranges[i], 2)) + ', ' + str(
            round(p_ranges[i + 1], 2)) + ']')
        plt.plot(bins[i], one_gaussians(bins[i], *optim[i][3 * pdg_code:3 * pdg_code + 3]), linewidth=2)
        plt.yscale('log')
        ax.set_facecolor('w')
        # plt.pause(0.1)
        plt.savefig((f'{Config.source_fp}/plots/p_vs_tpc_gauss/{model_name}_{particle_name}_gauss_2.png'))
        plt.clf()

    def get_weights(ranges_lengths):
        return [x/sum(ranges_lengths) for x in ranges_lengths]

    def weighted_mean(df, weights):
        return sum([metric*weight for metric, weight in zip(df, weights)])

    weights = get_weights(ranges_lengths)
    print(f'Average efficiencies list: {avg_efficiency}')
    return weighted_mean(avg_gauss_prob, weights), weighted_mean(avg_model_prob, weights), weighted_mean(avg_efficiency,
                                                                                                         weights), weighted_mean(
        avg_accuracy, weights), weighted_mean(avg_f1, weights)
    # return avg_gauss_prob, avg_model_prob, avg_efficiency, avg_accuracy
