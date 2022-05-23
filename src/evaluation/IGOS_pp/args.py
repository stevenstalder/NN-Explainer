import argparse


def init_args():

    parser = argparse.ArgumentParser(
        description='Generate explanations using I-GOS and iGOS++.'
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle the dataset.')

    parser.add_argument(
        '--size',
        type=int,
        default=28,
        help='The resolution of mask to be generated.')

    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='The input size to the network.')

    parser.add_argument(
        '--num_samples',
        type=int,
        default=5000,
        help='The number of samples to run explanation on.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='The number of images to generate explanations at once.')

    parser.add_argument(
        '--manual_seed',
        type=int,
        default=63,
        help='The manual seed for experiments.')

    parser.add_argument(
        '--L1',
        type=float,
        default=1
    )

    parser.add_argument(
        '--L2',
        type=float,
        default=20
    )

    parser.add_argument(
        '--ig_iter',
        type=int,
        default=20)

    parser.add_argument(
        '--iterations',
        type=int,
        default=15
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1000
    )

    return parser.parse_args()
