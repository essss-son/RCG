import torch

from gen_eval_util import sent_gen_eval
# from gen_eval_util.detoxification_gen_eval import detoxic_gen_eval
# from gen_eval_util.topic_gen_eval import topic_gen_eval
# ide
# from version_9.utils.Agent import Agent
#bash
from utils import Agent

import json
import argparse

def get_args(hyper_params_path):
    with open(hyper_params_path, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    return args

def get_agent(args, policy_path):
    agent = Agent(args)
    agent.load(policy_path)
    return agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, default=None)
    parser.add_argument('--policy_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_sequence', type=int, default=None)
    parser.add_argument('--generate_length', type=int, default=None)
    args1 = parser.parse_args()

    args_path = "/home/anke/DXZ/RCG/version_9/distil_test/rl_train/sentiment/pos/distil_test_2/hyper_params.json"
    policy_path = "/home/anke/DXZ/RCG/version_9/distil_test/rl_train/sentiment/pos/distil_test_2/checkpoints/Epoch50_ACC_96.7_ppl52.9"

    # args_path = args1.args_path
    # policy_path = args1.policy_path

    args = get_args(args_path)
    args.lora_path_dict = {
        "pos": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "neg": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/neg/epoch8_lora_acc0.766",
        "world": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "sports": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "business": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "science": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "toxic": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "nontoxic": "/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
    }
    # args.batch_size = args1.batch_size
    # args.num_sequence = args1.num_sequence
    # args.generate_length = args1.generate_length
    #
    # note eval hyperparameters PLZ set
    args.batch_size = 16
    args.num_sequence = 32
    args.generate_length = 64





    tag = policy_path.split('/')[-1][:7]+f"_length_{args.generate_length}"

    args.output_path = f"./eval_result/{args.task}/{args.attr}/{args.version}/{tag}/"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.agent = get_agent(args,policy_path)

    if args.task == "sentiment":
        sent_gen_eval(args)
    elif args.task == "topic":
        # topic_gen_eval(args)
        pass
    elif args.task == "detoxification":
        # detoxic_gen_eval(args)
        pass