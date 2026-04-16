import os
import sys
sys.path.append(os.path.dirname(__file__))
# print(__file__)
# print(sys.path)
import json
import random
import numpy as np
import torch
from utils import compute_distinct
from utils import Agent
from tqdm import tqdm

from utils import my_eval_set
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



@torch.no_grad()
def collect_trajectory(args,epoch):
    batch_size = args.collect_trj_batch_size
    collect_steps = args.collect_steps
    agent = args.agent
    ids_state = agent.reset(batch_size=batch_size)
    batch_ids = ids_state.input_ids

    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []
    batch_values = []

    for step in range(collect_steps):
        batch_states.append(batch_ids)
        states = batch_ids

        action, log_prob, value, kl_d = agent.get_action_log_prob(states)
        batch_ids, reward = agent.step(batch_ids, action, kl_d, step + collect_steps * epoch)
        batch_actions.append(action)
        batch_rewards.append(reward)
        batch_log_probs.append(log_prob)
        batch_values.append(value)
        # batch_kl.append(kl_d)

    return batch_states, batch_actions, batch_rewards, batch_log_probs,batch_values



@torch.no_grad()
def compute_gae(batch_reward, batch_value, gam=0.98, lam=0.98):
    gae = 0
    batch_done = torch.zeros_like(batch_reward)
    batch_done[:,-1] = torch.ones(batch_reward.shape[0],device=batch_reward.device)
    tail_value = torch.zeros(batch_value.shape[0], 1, device=batch_value.device)
    batch_value = torch.cat([batch_value, tail_value], dim=-1)
    advantage = torch.zeros_like(batch_reward)
    for i in reversed(range(batch_done.shape[1])):
        delta = batch_reward[:,i] + gam * batch_value[:,i + 1] * (1 - batch_done[:,i]) - batch_value[:,i]
        gae = delta + gam * lam * gae * (1 - batch_done[:,i])
        advantage[:,i] = gae
    returns = advantage + batch_value[:,:-1]
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return advantage.detach(), returns.detach()

def train(args):
    # return
    total_epoch = args.total_train_epoch
    eval_epoch = 0
    write_steps = 0
    max_acc = -1e4
    min_ppl = 1e4
    min_toxic = 1e4
    trj_path = f"./rl_train/{args.task}/{args.attr}/"
    ckpt_path = f"./rl_train/{args.task}/{args.attr}/" + args.version + "/checkpoints/"
    os.makedirs(trj_path,exist_ok=True)
    trj_text_path = trj_path + args.version
    os.makedirs(trj_text_path, exist_ok=True)

    for epoch in tqdm(range(total_epoch)):
        batch_states, batch_actions, batch_rewards, batch_log_probs, batch_values = collect_trajectory(args,epoch)
        with open(trj_text_path + "/collected_trj.txt", 'a') as f:
            texts = agent.tokenizer.batch_decode(batch_states[-1][:8,:])
            for text in texts:
                f.write(f"Epoch {epoch+1}/{total_epoch}:" + text.replace('\n',' ') + '\n')


        batch_rewards = torch.cat(batch_rewards, dim=-1)
        batch_values = torch.cat(batch_values, dim=-1)
        args.writer.add_scalar("Average reward", batch_rewards.mean(), epoch)
        advantage, returns = compute_gae(batch_rewards, batch_values)

        batch_adv = torch.split(advantage, 1, dim=-1)
        batch_return = torch.split(returns, 1, dim=-1)

        write_steps = agent.ppo_update(batch_states, batch_actions,batch_log_probs,batch_adv,batch_return,write_steps)


        if (epoch+1) % args.epochs_for_eval == 0:
            eval_texts = evaluate_texts(args)
            d1, d2, d3 = compute_distinct(eval_texts)
            dist_n = round((d1+d2+d3)/3, 3)
            ppl_list = evaluate_ppl(agent, eval_texts, args.eval_batch_size)
            m_ppl = round(sum(ppl_list)/len(ppl_list),2)
            acc = None
            acc_list = None
            if args.task == "sentiment":
                acc, acc_list = compute_sentiment_acc(agent, eval_texts, args.eval_batch_size, args.attr)
                args.writer.add_scalar("Evaluation Accuracy", acc, eval_epoch)
            elif args.task == "topic":
                acc, acc_list = compute_topic_acc(agent,eval_texts,args.eval_batch_size, args.attr)
                args.writer.add_scalar("Evaluation Accuracy", acc, eval_epoch)
            elif args.task == "detoxification":
                acc, acc_list = compute_toxic_acc(agent, eval_texts, args.eval_batch_size)
                args.writer.add_scalar("Evaluation toxicity", acc, eval_epoch)

            if acc is None:
                raise ValueError("No valid accuracy found")

            eval_epoch += 1
            text_output = f"./rl_train/{args.task}/{args.attr}/" + args.version + "/text_output/"
            os.makedirs(text_output,exist_ok=True)
            acc_map = {
                "sentiment":"ACC",
                "topic":"ACC",
                "detoxification":"TOX",
            }
            output_text_path = text_output + f"Epoch{(epoch+1):02d}_{acc_map[args.task]}_{acc:.2f}_ppl{m_ppl}_dist_{dist_n}.jsonl"
            write_output_text_to_file(eval_texts,output_text_path,ppl_list,args.task,acc_list)

            if args.task == "detoxification":
                # if acc < min_toxic:
                save_path = ckpt_path + f"Epoch{(epoch+1):02d}_Toxicity_{acc:.1f}_ppl{m_ppl:.1f}/"
                os.makedirs(save_path, exist_ok=True)
                max_acc = acc
                agent.save(save_path)
            else:
                # if acc > max_acc:
                save_path = ckpt_path + f"Epoch{(epoch+1):02d}_ACC_{acc:.1f}_ppl{m_ppl:.1f}/"
                os.makedirs(save_path, exist_ok=True)
                max_acc = acc
                agent.save(save_path)
@torch.no_grad()
def evaluate_texts(args):

    num_sequence = args.num_sequence
    generate_length = args.generate_length
    collect_batch_size = args.collect_batch_size
    agent = args.agent
    collect_epoch = num_sequence // collect_batch_size
    all_texts = []
    prompts = agent.reset(batch_size=collect_batch_size,generation=True)
    for prompt in prompts:
        input_ids = agent.tokenizer.encode(prompt, return_tensors='pt').to(agent.device)
        init_batch_ids = input_ids.expand(collect_batch_size, -1)
        single_text_for_prompt = []
        for round in range(collect_epoch):
            batch_ids = init_batch_ids
            for step in range(generate_length):
                action, _ = agent.get_action_log_prob(batch_ids, evaluate=True)
                batch_ids, _ = agent.step(batch_ids, action, _ ,evaluate=True)
            texts = agent.tokenizer.batch_decode(batch_ids)
            generate_text = [text.strip('\n').replace('\n', ' ') for text in texts]
            single_text_for_prompt.extend(generate_text)
        all_texts.extend(single_text_for_prompt)

    return all_texts
@torch.no_grad()
def evaluate_ppl(agent, texts, eval_batch_size):
    tokenizer = agent.tokenizer
    model = agent.ref_gpt2
    ppl_list = []
    my_data = my_eval_set(texts)
    my_loader = DataLoader(my_data, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    for batch in my_loader:
        encoded = tokenizer(batch,padding=True,return_tensors="pt").to(agent.device)
        shifted_mask = encoded['attention_mask'][:,1:]
        true_num = shifted_mask.sum(dim=-1,keepdim=True)
        logits = model(**encoded).logits
        probs = logits[:,:-1,:].log_softmax(dim=-1)
        target = encoded['input_ids'][:,1:]
        probs = probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1) * shifted_mask
        ppl_ = -(probs.sum(dim=-1,keepdim=True) / true_num)
        ppl = torch.exp(ppl_)
        ppl_list.append(ppl)
    ppl = torch.cat(ppl_list, dim=0).tolist()
    ppl = [x[0] for x in ppl]
    return ppl

@torch.no_grad()
def compute_sentiment_acc(agent, texts, eval_batch_size, attr):
    eval_data = my_eval_set(texts)
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size,shuffle=False,drop_last=False)
    pos_num = 0
    neg_num = 0
    acc_list = []
    for batch in eval_loader:
        encoded = agent.tokenizer(batch,return_tensors='pt',padding=True).to(agent.device)
        logits = agent.reward_model(**encoded).logits
        probs = F.softmax(logits,dim=-1)
        batch_acc = probs.argmax(dim=-1).tolist()
        acc_list.extend(batch_acc)

        pos_preds = torch.argmax(probs,dim=-1).sum().item()
        neg_preds = len(batch) - pos_preds

        neg_num += neg_preds
        pos_num += pos_preds

    pos_acc = round(pos_num / len(eval_data),4) * 100
    neg_acc = round(neg_num / len(eval_data),4) * 100
    if attr == "pos":
        return pos_acc,acc_list
    elif attr == "neg":
        return neg_acc,acc_list



@torch.no_grad()
def compute_topic_acc(agent, texts, eval_batch_size, attr):
    model = agent.reward_model
    tokenizer = agent.reward_tokenizer
    attr_dict = {
        "world":0,
        "sports":1,
        "business":2,
        "science":3,
    }
    eval_set = my_eval_set(texts)
    dataloader = DataLoader(eval_set, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    correct = 0
    label_list = []
    for batch in dataloader:
        encoded = tokenizer(batch,padding=True,return_tensors='pt',truncation=True,max_length=512).to(agent.device)
        output = model(**encoded).logits.softmax(dim=-1).argmax(-1).tolist()

        c_list = [1 if attr_dict[attr] == x else 0 for x in output]
        correct += sum(c_list)
        label_list.extend(output)
        # print(c_list)
    acc = round(correct / len(eval_set) * 100, 2)
    return acc, label_list


@torch.no_grad()
def compute_toxic_acc(agent, texts, eval_batch_size):
    model = agent.reward_model
    tokenizer = agent.reward_tokenizer
    eval_set = my_eval_set(texts)
    loader = DataLoader(eval_set, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    total_toxic = []
    for batch in loader:
        input = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(agent.device)
        outputs = model(**input)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=-1)
        tox_id = model.config.label2id["toxicity"]
        tox_scores = probs[:, tox_id].tolist()  # shape: [B]
        total_toxic.extend(tox_scores)
    ave_toxic = round(sum(total_toxic) / len(total_toxic) * 100, 2)
    return ave_toxic, total_toxic



def write_output_text_to_file(eval_texts,write_path,ppl_list,task,acc_list=None):
    with open(write_path, "w", encoding='utf-8') as f:
        if task == "topic":
            eval_labels = {
                0: "world",
                1: "sports",
                2: "business",
                3: "science"
            }
            for text, ppl, acc in zip(eval_texts, ppl_list,acc_list):
                tattr = eval_labels[acc]
                obj = {"text": text, "ppl": ppl, "topic": tattr}
                json.dump(obj, f, ensure_ascii=False)
                f.write('\n')
        elif task == "sentiment":
            for text, ppl,s in zip(eval_texts, ppl_list,acc_list):
                sent = "positive" if s == 1 else "negative"
                obj = {"text": text, "ppl": ppl, "sentiment": sent}
                json.dump(obj, f, ensure_ascii=False)
                f.write('\n')
        else:
            for text, ppl,t in zip(eval_texts, ppl_list,acc_list):
                obj = {"text": text, "ppl": ppl, "toxicity": t}
                json.dump(obj, f, ensure_ascii=False)
                f.write('\n')


def save_hyper_params(args,save_path):
    TRAIN_EVAL_KEYS = [
        # train
        "total_train_epoch",
        "collect_trj_batch_size",
        "collect_steps",
        "critic_lora_r",
        "critic_lora_alpha",
        "ppo_update_epoch",
        "update_batch_size",
        "actor_lr",
        "critic_lr",
        "TEMPERATURE",
        "TOPK",
        "NOISE_SCALING",
        # reward
        "reward_scaling",
        "base_probs_scaling",
        "repeat_scaling",

        # evaluate
        "epochs_for_eval",
        "num_sequence",
        "generate_length",
        "eval_batch_size",
        "collect_batch_size",

        # meta
        "task",
        "attr",
        "version",
    ]

    os.makedirs(save_path, exist_ok=True)
    cfg = {k: getattr(args, k) for k in TRAIN_EVAL_KEYS}
    with open(os.path.join(save_path, "hyper_params.json"), "w") as f:
        json.dump(cfg, f, indent=2)



if __name__ == '__main__':
    num_s_dict = {
        "sentiment":8,
        "topic":8,
        "detoxification":1
    }
    collect_b_dict = {
        "sentiment": 8,
        "topic": 8,
        "detoxification": 1
    }

    # set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sentiment", help="sentiment ot topic or detoxification")
    parser.add_argument("--attr", type=str, default="pos",help="pos neg or world sports business science or nontoxic")
    parser.add_argument("--version",type=str,default="distil_test_2")
    # parser.add_argument("--base_probs_scaling",type=float,default=None)

    args = parser.parse_args()
    print("current version is {}".format(args.version))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.lora_path_dict = {
        "pos":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "neg":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/neg/epoch8_lora_acc0.766",
        "world":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "sports":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "business":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "science":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "toxic":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
        "nontoxic":"/home/anke/DXZ/RCG/version_9/distil_test/lora_train/sentiment/lora_ckpt/c_attn_lora/pos/epoch8_lora_acc0.82",
    }

    # note train
    args.total_train_epoch = 50
    args.collect_trj_batch_size = 32
    args.collect_steps = 64
    args.critic_lora_r = 16
    args.critic_lora_alpha = 32
    args.ppo_update_epoch = 1
    args.update_batch_size = 64
    args.actor_lr = 3e-6
    args.critic_lr = 2e-5
    args.TEMPERATURE = 1.0
    args.TOPK = 200
    args.NOISE_SCALING = 1

    args.NOISE_DECAY = 0

    #note reward
    args.reward_scaling = 2.0
    args.base_probs_scaling = 0.08
    args.repeat_scaling = 0.0

    # note evaluate
    args.epochs_for_eval = 10
    args.num_sequence = num_s_dict[args.task]     #for each task  8 8 1
    args.collect_batch_size = collect_b_dict[args.task]  # for each task 8 8 1
    args.generate_length = 64
    args.eval_batch_size = 32


    # note record
    writer = SummaryWriter(log_dir=f"./rl_train/{args.task}/{args.attr}/" + args.version + "/logs/")
    args.writer = writer
    hyp_path = os.path.join(f"./rl_train/{args.task}/{args.attr}", args.version)
    save_hyper_params(args,hyp_path)

    agent = Agent(args)
    args.agent = agent
    train(args)
