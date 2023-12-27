import argparse
import parses.parses as parses
from test.evaluator import name2evaluator
import time

start = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_iter',
    default=1000,
    type=int,
    help='ransac iterations')
parser.add_argument(
    '--dataset',
    default='CTCS',
    type=str,
    help='dataset name')
parser.add_argument(
    '--ransac_d',
    default=-1,
    type=float,
    help='inliner threshold of ransac')
parser.add_argument(
    '--tau',
    default=2,
    type=float,
    help='tau for FMR')

args = parser.parse_args()

config,nouse=parses.get_config()
config.ok_match_dist_threshold=args.tau
if args.ransac_d>0:
    config.ransac_c_inlinerdist=args.ransac_d
config.testset_name=args.dataset
config.weight = True
eval_net=name2evaluator[config.evaluator](config,max_iter=args.max_iter)
eval_net.eval()

end = time.perf_counter()
runtime = end-start
msg = f"Disparity weight : {config.weight}" + '\n' "TestTime: "+str(runtime)+"s"+'\n'
with open('data/results.log','a') as f:
    f.write(msg+'\n')
print(msg)