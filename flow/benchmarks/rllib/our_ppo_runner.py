"""Runs the environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import argparse

import ray

from examples.rllib.multiagent_exps.multiagent_bottleneck import setup_exps

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments

EXAMPLE_USAGE = """
example usage:
    python ppo_runner.py grid0
Here the arguments are:
benchmark_name - name of the benchmark to run
num_rollouts - number of rollouts to train across
num_cpus - number of cpus to use for training
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--upload_dir", type=str, help="S3 Bucket to upload to.")

parser.add_argument(
    "--checkpoint", type=str, help="The checkpoint to load.")

# required input parameters
parser.add_argument(
    "--benchmark_name", type=str, help="File path to solution environment.")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=50,
    help="The number of rollouts to train over.")

# optional input parameters
parser.add_argument(
    '--num_cpus',
    type=int,
    default=2,
    help="The number of cpus to use.")

if __name__ == "__main__":
    benchmark_name = 'our_bottleneck'
    args = parser.parse_args()
    # benchmark name
    benchmark_name = args.benchmark_name
    # number of rollouts per training iteration
    num_rollouts = args.num_rollouts
    # number of parallel workers
    num_cpus = args.num_cpus

    upload_dir = args.upload_dir
    checkpoint = args.checkpoint

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # initialize a ray instance
    ray.init(num_cpus=8)
    alg_run, env_name, config = setup_exps(flow_params, evaluate=True)
    config['num_workers'] = num_cpus
    config['train_batch_size'] = 1000 * num_rollouts  # 1000 is the horizon
    config['lr'] = 0  # Stop learning

    exp_tag = {
        "run": alg_run,
        "env": env_name,
        "config": {
            **config
        },
        "checkpoint_freq": 25,
        "max_failures": 999,
        "stop": {
            "training_iteration": 500
        },
        "num_samples": 3,
        **({"restore": checkpoint} if checkpoint else {})
    }

    if upload_dir:
        exp_tag["upload_dir"] = "s3://" + upload_dir

    trials = run_experiments({
        flow_params["exp_tag"]: exp_tag
    })


# python our_ppo_runner.py --checkpoint /home/gitaar9/AI/DMAS_RESULTS/PPO_BottleneckMultiAgentEnvFinal-v0_0_2019-10-24_new_2_av_perc_50/checkpoint_160/checkpoint-160 --benchmark_name our_bottleneck --num_rollouts 8
# python our_ppo_runner.py --checkpoint /home/gitaar9/AI/DMAS_RESULTS/PPO_BottleneckMultiAgentEnvFinal-v0_0_2019-10-21_new_1/checkpoint_150/checkpoint-150 --benchmark_name our_bottleneck_10_perc_av --num_rollouts 8