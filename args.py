from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./data/processed/', help='Data directory')
parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')

args = parser.parse_args()