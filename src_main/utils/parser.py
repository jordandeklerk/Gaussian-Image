import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('CS 444 UIUC MP2', add_help=False)
    parser.add_argument('--image_url', default='https://bmild.github.io/fourfeat/img/lion_orig.png', type=str)
    parser.add_argument('--mapping_size', default=256, type=int)
    parser.add_argument('--scale', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument("--warmup_epochs", default=50, type=int)
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--input_channels', default=2, type=int)
    parser.add_argument('--output_channels', default=3, type=int)
    parser.add_argument('--display_epochs', default=[0, 100, 400, 900, 1000], type=list)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--crop_center', default=512, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--seed', default=315, type=int,
                        help='seed')
    return parser

args, unknown = get_args_parser().parse_known_args()