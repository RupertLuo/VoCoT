from argparse import ArgumentParser
import os
from utils.logger import setup_logger
import json
import torch

def rank0_print(args, res):
    if args.local_rank==0 or args.local_rank == -1:
        print(res)

def get_output_name(args, mid_output=True):
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}.json'.format(args.dataset_name))

def get_all_output_names(args):
    return [os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, r)) for r in range(args.n_gpus)]



def main():
    parser = ArgumentParser()
    parser.add_argument('--config_arg', type=str, default=None)
    old_args = parser.parse_args()

    args = torch.load(old_args.config_arg)
    print(args)

    base_config_name = os.path.basename(args.eval_data)
    args.dataset_name = base_config_name[:-5] if base_config_name.endswith('.yaml') else base_config_name


    full_res = []
    for fn in get_all_output_names(args):
        full_res.extend(json.load(open(fn, 'r')))
        os.remove(fn)
    with open(get_output_name(args, mid_output=False), 'w') as wf:
        json.dump(full_res, wf)
    # saving the arguments
    torch.save(args, get_output_name(args, mid_output=False)[:-4]+'args.bin')
            

if __name__=='__main__':
    main()