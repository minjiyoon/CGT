import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    # GNN training-related hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch.')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of workers for data loader.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of precetched batchs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait before early stop.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")

    # Dataset-related hyperparameters
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Dataset location.')
    parser.add_argument('--save_dir', type=str, default="save",
                        help='Save location.')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')

    # GNN structure-related hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--step_num', type=int, default=2,
                        help='Number of propagating steps')
    parser.add_argument('--sample_num', type=int, default=5,
                        help='Number of sampled neighbors')
    parser.add_argument('--subgraph_step_num', type=int, default=2,
                        help='Number of propagating steps')
    parser.add_argument('--subgraph_sample_num', type=int, default=5,
                        help='Number of propagating steps')

    # Privacy-related hyperparameters
    parser.add_argument('--dp_feature', dest='dp_feature', action='store_true')
    parser.set_defaults(dp_feature=False)
    parser.add_argument('--dp_edge', dest='dp_edge', action='store_true')
    parser.set_defaults(dp_edge=False)
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--dp-sigma",
        type=float,
        default=0.01,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--dp-max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--dp_delta",
        type=float,
        default=100,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--dp_epsilon",
        type=float,
        default=100.,
        metavar="D",
        help="Target epsilon",
    )

    # SR-GNN-related hyperparameters
    parser.add_argument("--arch", type=int, default=0,
                        help="use which variant of the model")
    parser.add_argument("--alpha", type=float, default=0.,
                        help="restart coefficient in biased sampling")
    parser.add_argument('--iid_sample', dest='iid_sample', action='store_true')
    parser.add_argument('--bias_sample', dest='iid_sample', action='store_false')
    parser.set_defaults(iid_sample=True)

    # Computation Graph encoding-related hyperparameters
    parser.add_argument('--org_code', dest='org_code', action='store_true')
    parser.add_argument('--dup_code', dest='org_code', action='store_false')
    parser.set_defaults(org_code=True)
    parser.add_argument('--self_connection', dest='self_connection', action='store_true')
    parser.set_defaults(self_connection=True)

    # Task-related hyperparameters
    parser.add_argument('--task_name', type=str, default="aggregation",
                        help='aggregation, depth, shift, width')
    parser.add_argument('-n', '--model_list', nargs='+', default=['gcn', 'sgc', 'gin', 'gat'],
                        help='a list of GNN models to be tested')
    parser.add_argument('-p', '--predictor_list', nargs='+', default=['dot', 'mlp'],
                        help='a list of link predictor models to be tested')
    parser.add_argument("--noise_num", type=int, default=0,
                        help="Number of noise edges")

    # Cluster-related hyperparameters
    parser.add_argument('--cluster_num', type=int, default=512,
                        help='Number of clusters used to discretize feature vectors')
    parser.add_argument('--cluster_size', type=int, default=1,
                        help='Size of mininum cluster')
    parser.add_argument('--cluster_sample_num', type=int, default=5000,
                        help='Number of nodes participated in kmeans')

    # GPT-related hyperparameters
    parser.add_argument('--gpt_train_name', type=str, default="default",
                        help='wandb run name')
    parser.add_argument('--gpt_model', type=str, default="XLNet",
                        help='GPT, XLNet, or Bayes')
    parser.add_argument('--gpt_softmax_temperature', type=float, default=1.,
                        help='Temperature used to sample')
    parser.add_argument('--gpt_epochs', type=int, default=5,
                        help='Number of epochs to train.')
    parser.add_argument('--gpt_batch_size', type=int, default=128,
                        help='Size of batch.')
    parser.add_argument('--gpt_lr', type=float, default=0.003/8,
                        help='Initial learning rate.')
    parser.add_argument('--gpt_layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--gpt_heads', type=int, default=12,
                        help='Number of heads')
    parser.add_argument('--gpt_dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gpt_weight_decay', type=float, default=5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--gpt_hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--gpt_early_stopping', type=int, default=10,
                        help='Number of epochs to wait before early stop.')

    # CGT-related hyperparameters
    parser.add_argument('--give_start_id', dest='gpt_start_id', action='store_true')
    parser.set_defaults(gpt_start_id=False)
    parser.add_argument('--no_label_con', dest='gpt_label_con', action='store_false')
    parser.set_defaults(gpt_label_con=True)
    parser.add_argument('--long_seq', dest='gpt_long_seq', action='store_true')
    parser.set_defaults(gpt_long_seq=False)
    parser.add_argument('--inv_pos', dest='gpt_inv_pos', action='store_true')
    parser.set_defaults(gpt_inv_pos=False)

    parser.add_argument('--label_wise', dest='label_wise', action='store_true')
    parser.add_argument('--no_label_wise', dest='label_wise', action='store_false')
    parser.set_defaults(label_wise=True)

    # Save intermediate graph infomation
    parser.set_defaults(save_org_graph=False)
    parser.set_defaults(save_cluster_graph=False)
    parser.set_defaults(save_synthetic_graph=False)


    args, _ = parser.parse_known_args()
    return args
