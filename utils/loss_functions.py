from utils.datasets import TargetDataset
from tools.utils import ForeverDataIterator
import torch
from utils.config import Config
from domain_adaptation.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from domain_adaptation.modules.kernels import GaussianKernel


mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=not Config.dan_non_linear, quadratic_program=Config.dan_quadratic_program
        )


def get_mean_loss(loss_function, df_source, df_target):
    """
    Calculates mean loss using certain loss function
    between source and target datasets.
    If any of datasets is empty, returns None
    """
    losses_list = []

    source_dataset = TargetDataset(df_source)
    target_dataset = TargetDataset(df_target)

    source_loader = torch.utils.data.DataLoader(source_dataset, Config.batch_size)
    target_loader = torch.utils.data.DataLoader(target_dataset, Config.batch_size)

    train_source_iter = ForeverDataIterator(source_loader)
    train_target_iter = ForeverDataIterator(target_loader)

    range_len = min(len(source_loader), len(target_loader))

    for i, _ in enumerate(range(range_len)):
        if i % 10 == 0:
            print(f'{i} / {range_len}')

        source_x, _ = next(train_source_iter)
        target_x, _ = next(train_target_iter)

        min_vector_len = min(len(source_x), len(target_x))
        if min_vector_len > 40:
            source_x = source_x[:min_vector_len]
            target_x = target_x[:min_vector_len]

            loss = loss_function(source_x, target_x)

            losses_list.append(abs(loss.item()))

    if len(losses_list) == 0:
        return None

    return sum(losses_list) / len(losses_list)
