import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    # Training-related hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch.')
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

    # GCN structure-related hyperparameters
    parser.add_argument('--task_name', type=str, default="aggregation",
                        help='aggregation, depth, shift, width')
    parser.add_argument('-n', '--model_list', nargs='+', default=['gcn', 'gat', 'sgc', 'gin'])
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

    # Adjacency coding way
    parser.add_argument('--org_code', dest='org_code', action='store_true')
    parser.add_argument('--dup_code', dest='org_code', action='store_false')
    parser.set_defaults(org_code=True)
    parser.add_argument('--self_connection', dest='self_connection', action='store_true')
    parser.set_defaults(self_connection=False)

    # Task
    parser.add_argument("--noise_num", type=int, default=0,
                        help="Number of noise edges")

    # Cluster-related hyperparameters
    parser.add_argument('--cluster_num', type=int, default=512,
                        help='Number of clusters used to discretize feature vectors')
    parser.add_argument('--cluster_size', type=int, default=2,
                        help='Size of mininum cluster')
    parser.add_argument('--cluster_sample_num', type=int, default=5000,
                        help='Number of nodes participated in kmeans')

    # GPT-related hyperparameters
    parser.add_argument('--gpt_train_name', type=str, default="default",
                        help='run name')
    parser.add_argument('--gpt_model', type=str, default="XLNet",
                        help='XLNet')
    parser.add_argument('--gpt_softmax_temperature', type=float, default=1.,
                        help='Temperature used to sample')
    parser.add_argument('--gpt_epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--gpt_batch_size', type=int, default=128//8,
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

    parser.add_argument('--label_wise', dest='label_wise', action='store_true')
    parser.add_argument('--no_label_wise', dest='label_wise', action='store_false')
    parser.set_defaults(label_wise=True)


    args, _ = parser.parse_known_args()
    return args
