class Config:

    # preprocessing
    columns_to_drop_during_preprocessing = ['fTOFsignalRaw', 'fTPCsignalN', 'fTPCnclsIter1', 'fTPCnclsFIter1',
                                            'fTPCnclsF', 'fTRDQuality']

    particles_dict = {
        0: 'electrons',
        1: 'pions',
        2: 'kaons',
        3: 'protons'
    }

    da_models_list = ['cdan', 'dan', 'jan', 'dann', 'mdd', 'wdgrl']
    # da_models_list = ['dan']
    particles_list = ['electrons', 'pions', 'kaons', 'protons']
    # particles_list = ['kaons']

    # file paths
    source_fp = f'../data'
    training_data_fp = f'{source_fp}/pickles/training_data/'
    raw_data_prod_fp = f'{training_data_fp}/production.pkl'
    raw_data_sim_fp = f'{training_data_fp}/simulation.pkl'
    preprocessed_data_prod_fp = f'{training_data_fp}/data_prod_preprocessed.pkl'
    preprocessed_data_sim_fp = f'{training_data_fp}/data_sim_preprocessed.pkl'
    models_dir = f'{source_fp}/trained_models/'
    source_model_fp = f'{source_fp}/trained_models/source'
    cdan_model_fp = models_dir + 'cdan'
    dan_model_fp = models_dir + 'dan'
    jan_model_fp = models_dir + 'jan'
    dann_model_fp = models_dir + 'dann'
    mdd_model_fp = models_dir + 'mdd'
    wdgrl_model_fp = models_dir + 'wdgrl'
    xgb_model_fp = models_dir + 'xgb'
    cb_model_fp = models_dir + 'cb'
    random_forest_model_fp = models_dir + 'random_forest'

    # used for all attributes expects 4 (its_signal) and 6 (tpc_signal)
    perturbation = 0.05
    # used for perturbating attributes 4 (its_signal) and 6 (tpc_signal)
    # which has distribution similar to normal and needs more perturbation
    its_signal_perturbation = 0.00
    tpc_signal_perturbation = 0.00

    # adaptation quality
    max_score = 100
    parts = 30
    min_ratio = 2

    # should train certain model
    train_source = True
    train_cdan = True
    train_dan = True
    train_dann = True
    train_jan = True
    train_mcd = True
    train_mdd = True
    train_wdgrl = True

    """
    MODEL TESTING
    """
    plot_umap = True
    validate = True
    # distance_matrix = True
    print_classification_report = True

    """
    GENERAL
    """
    n_features = 6
    n_classes = 2
    batch_size = 512  # mini-batch size (default: 512)
    frac = 0.5
    seed = 0

    # if we test on perturbed dataset we set test size as 0.3
    test_size = 0.3

    # if we test on production dataset we want the training part
    # of the source dataset to be the same size as production dataset
    # 26497 - len of prod dataset, 541572 - len of source dataset
    # test_size = 26497 / (541572 * frac)

    """
    SOURCE
    """
    net_out_features = 200
    net_hidden_dim = 200
    source_workers = 2  # number of data loading workers (default: 4)
    source_epochs = 15  # number of total epochs to run'
    source_lr = 0.005  # initial learning rate (default: 0.1)
    source_momentum = 0.9  # momentum
    source_weight_decay = 0.001  # weight decay (default: 1e-3)
    source_print_freq = 100  # print frequency (default: 100)
    source_epochs_dict = 30

    """
    CDAN
    """
    cdan_workers = 2  # number of data loading workers (default: 4)
    cdan_lr = 0.01  # initial learning rate (default: 0.1)
    cdan_momentum = 0.9  # momentum
    cdan_weight_decay = 1e-3  # weight decay (default: 1e-3)
    cdan_print_freq = 100  # print frequency (default: 100)
    cdan_trade_off = 5.  # the trade-off hyper-parameter for transfer loss (default: 1)
    cdan_entropy = False  # use entropy conditioning
    cdan_epochs = 20

    """
    DAN
    """
    dan_workers = 2  # number of data loading workers (default: 4)
    dan_epochs = 1  # number of total epochs to run'
    dan_lr = 0.003  # initial learning rate (default: 0.003)
    dan_momentum = 0.9  # momentum
    dan_weight_decay = 0.0005  # weight decay (default: 0.0005)
    dan_print_freq = 100  # print frequency (default: 100)
    dan_iters_per_epoch = 1  # number of iterations per epoch
    dan_trade_off = 1.  # the trade-off hyper-parameter for transfer loss (default: 1)
    dan_non_linear = False  # whether not use the linear version
    dan_quadratic_program = False  # whether use quadratic program to solve beta
    dan_epochs = 10

    """
    JAN
    """
    jan_workers = 2  # number of data loading workers (default: 4)
    jan_epochs = 1  # number of total epochs to run'
    jan_lr = 0.003  # initial learning rate (default: 0.003)
    jan_momentum = 0.9  # momentum
    jan_weight_decay = 0.0005  # weight decay (default: 0.0005)
    jan_print_freq = 100  # print frequency (default: 100)
    jan_trade_off = 1.  # the trade-off hyper-parameter for transfer loss (default: 1)
    jan_linear = False  # whether use the linear version
    jan_adversarial = False  # whether use adversarial theta
    jan_epochs = 5

    """
    MCD
    """
    mcd_workers = 2  # number of data loading workers (default: 4)
    mcd_lr = 0.001  # initial learning rate (default: 0.001)
    mcd_num_k = 4  # how many steps to repeat the generator update
    mcd_print_freq = 100  # print frequency (default: 100)
    mcd_trade_off = 1.  # the trade-off hyper-parameter for transfer loss (default: 1)
    mcd_bottleneck_dim = 200
    mcd_center_crop = False
    mcd_epochs = 20

    """
    DANN
    """
    dann_lr = 0.01
    dann_momentum = 0.9
    dann_weight_decay = 1e-3
    dann_print_freq = 100
    dann_trade_off = 1
    dann_workers = 2
    dann_epochs = 20

    """
    MDD
    """
    mdd_workers = 2  # number of data loading workers (default: 4)
    mdd_lr = 0.004  # initial learning rate
    mdd_momentum = 0.9  # momentum
    mdd_weight_decay = 0.0005  # weight decay (default: 5e-4)
    mdd_print_freq = 100  # print frequency (default: 100)
    mdd_margin = 4.  # margin gamma, (default: 4.)
    mdd_bottleneck_dim = 200  # (default: 1024)
    mdd_center_crop = False  # store_true
    mdd_trade_off = 1.  # the trade-off hyper-parameter for transfer loss (default: 1)
    mdd_lr_gamma = 0.0002  # (default: 0.0002)
    mdd_epochs = 20

    """
    WDGRL
    """
    wdgrl_k_critic = 5
    wdgrl_k_clf = 1
    wdgrl_gamma = 10
    wdgrl_wd_clf = 1
    wdgrl_print_freq = 1
    wdgrl_epochs = 10



