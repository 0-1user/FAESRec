import argparse

from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FAESRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Baby', help='name of datasets')

    args, _ = parser.parse_known_args()
    # Config files
    args.config_file_list = [
        'recbole/properties/overall.yaml',
        f'recbole/properties/model/{args.model}.yaml',
        f'recbole/properties/dataset/{args.dataset}.yaml'
    ]
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=args.config_file_list)
