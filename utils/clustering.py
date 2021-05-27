import pandas as pd
import numpy as np
from my_utils.minisom import MiniSom


def som_predict(x, som) -> int:
    """
    Predicts cluster basing on a data row and the model.
    Arguments:
        x: data row
        som: model
    Returns:
        cluster: number
    """
    result = som.winner(np.array(x))
    cluster = str(result[0]) + str(result[1])
    return int(cluster)


def cluster_df(som_width: int, som_height: int, df: pd.core.frame.DataFrame, n_iter: int, sigma=0.3,
              learning_rate=0.001):
    """
    Decided to use SOM clustering as it is robust to highly dimensional data,
    doesn't require too much RAM and is very fast.
    """

    som = MiniSom(som_width, som_height, df.shape[1], sigma=sigma, learning_rate=learning_rate,
                  random_seed=0)
    print('training som')
    som.train(df, n_iter)

    # converting numpy arrays to dataframes
    df = pd.DataFrame(df)
    print('assigning clusters')
    # creating column with cluster basing on model prediction
    df['cluster'] = df.apply(lambda x: som_predict(x, som), axis=1)

    return df
