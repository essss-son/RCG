import torch
from tqdm import trange

#note ide
# from version_9.utils.Agent import Agent
#note bash
from utils.Agent import Agent

import argparse
#note ide
# from version_9.rl_train import compute_sentiment_acc, evaluate_ppl
#note bash
# from ..rl_train import compute_sentiment_acc, evaluate_ppl

import os
import json

from .utils import compute_distinct, generation, compute_ppl, compute_sent_acc
@torch.no_grad()
def sent_gen_eval(args):
    output_path = args.output_path
    batch_size = args.batch_size
    os.makedirs(output_path, exist_ok=True)
    all_texts = generation(args)
    print("Evaluating...")

    ave_acc, acc_list = compute_sent_acc(args,all_texts, batch_size, args.attr)
    ave_ppl, ppl_list = compute_ppl(args,all_texts, batch_size)
    dist1, dist2, dist3 = compute_distinct(all_texts)

    # attr_dict = {
    #     "pos":"positive",
    #     "neg":"negative",
    # }
    file_name = f"{args.task}_{args.attr}_ACC_{ave_acc}_ppl_{ave_ppl}.jsonl"
    result_name = f"{args.task}_{args.attr}_result.json"
    with open(output_path + file_name, "w") as f:
        for text, ppl, attr in zip(all_texts, ppl_list, acc_list):
            s = "positive" if attr == 1 else "negative"
            obj = {"text":text, "ppl":ppl, "attr":s}
            json.dump(obj, f)
            f.write('\n')

    result = {
        "num_sequence":args.num_sequence,
        "length":args.generate_length,

        "task":args.task,
        "attr":args.attr,
        "acc":ave_acc,
        "ppl":ave_ppl,

        "dist-1":round(dist1,4),
        "dist-2":round(dist2,4),
        "dist-3":round(dist3,4),
    }
    with open(output_path + result_name, "w") as f:
        json.dump(result, f, indent=2)
    print("Finished, Eval result has been saved in", output_path)





def main():
    generation_length = 512
    generation_batch = 16
    total_sequences = 256
    eval_result_path = './eval_results/sentiment/'
    os.makedirs(eval_result_path, exist_ok=True)
    generation_epoch = total_sequences // generation_batch
    all_texts = []

    for i in trange(generation_epoch):
        batch_ids = agent.reset(generation_batch).input_ids

        for step in range(generation_length):
            action, _ = agent.get_action_log_prob(batch_ids, evaluate=True)
            batch_ids, _ = agent.step(batch_ids, action, _)
        texts = agent.tokenizer.batch_decode(batch_ids)
        generate_text = [text.strip('\n').replace('\n', ' ') for text in texts]
        all_texts.extend(generate_text)

    acc, acc_list = compute_sentiment_acc(agent, all_texts, generation_batch)
    ppl_list = evaluate_ppl(agent, all_texts, generation_batch)
    ave_ppl = round(sum(ppl_list)/len(ppl_list), 1)
    result_file = eval_result_path + f'14acc_{acc}_ppl{ave_ppl}_length_{generation_length}.jsonl'
    with open(result_file, 'w') as f:
        for text, ppl, ac in zip(all_texts, ppl_list, acc_list):
            sent = "positive" if ac == 1 else "negative"
            obj = {"text": text, "ppl": ppl, "sentiment": sent}
            json.dump(obj, f, ensure_ascii=False)
            f.write('\n')

def get_args(hyper_params_path):
    with open(hyper_params_path, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    return args

if __name__ == '__main__':

    hyper_params = "/home/anke/DXZ/freectrl/version_9/rl_train/sentiment/pos/v1/hyper_params.json"
    args = get_args(hyper_params)

    args.lora_path_dict = {
        "pos": "/home/anke/DXZ/freectrl/version_9/lora_train/sentiment/lora_ckpt/c_attn_lora/pos_lora_acc0.703",
        "neg": "/home/anke/DXZ/freectrl/version_9/lora_train/sentiment/lora_ckpt/c_attn_lora/neg_lora_acc0.75",
        "world": "/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/world/lora_acc82.81",
        "sports": "/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/sports/lora_acc87.5",
        "business": "/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/business/lora_acc64.06",
        "science": "/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/science/lora_acc92.19",
        "toxic": "/home/anke/DXZ/freectrl/version_9/lora_train/detoxification/lora_ckpt/c_attn_lora/toxic/epoch14_lora_toxic91.84",
        "nontoxic": "/home/anke/DXZ/freectrl/version_9/lora_train/detoxification/lora_ckpt/c_attn_lora/nontoxic/epoch14_lora_toxic22.93",
    }
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/anke/DXZ/freectrl/version_9/rl_train/v6.4/checkpoints/epoch9_sent_91.4_ppl28.5"

    agent = Agent(args)
    agent.load(model_path)

    print('load model successfully!')
    # main()