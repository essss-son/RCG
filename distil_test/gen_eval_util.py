import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from utils import Agent
import torch
def compute_distinct(texts):
    unique_1 = set()
    total_1 = 0

    unique_2 = set()
    total_2 = 0

    unique_3 = set()
    total_3 = 0

    for text in texts:
        tokens = text.strip().split()
        if not tokens: continue

        # Dist-1
        total_1 += len(tokens)
        unique_1.update(tokens)

        # Dist-2
        if len(tokens) >= 2:
            bigrams = list(zip(tokens, tokens[1:]))
            total_2 += len(bigrams)
            unique_2.update(bigrams)

        # Dist-3
        if len(tokens) >= 3:
            trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
            total_3 += len(trigrams)
            unique_3.update(trigrams)

    d1 = len(unique_1) / total_1 if total_1 > 0 else 0.0
    d2 = len(unique_2) / total_2 if total_2 > 0 else 0.0
    d3 = len(unique_3) / total_3 if total_3 > 0 else 0.0

    return d1, d2, d3


class my_eval_set(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

@torch.no_grad()
def compute_sent_acc(args, texts, eval_batch_size, attr):
    agent = args.agent
    eval_data = my_eval_set(texts)
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size,shuffle=False,drop_last=False)
    pos_num = 0
    neg_num = 0
    acc_list = []
    for batch in eval_loader:
        encoded = agent.reward_tokenizer(batch,return_tensors='pt',padding=True).to(agent.device)
        logits = agent.reward_model(**encoded).logits
        probs = F.softmax(logits,dim=-1)
        batch_acc = probs.argmax(dim=-1).tolist()
        acc_list.extend(batch_acc)

        pos_preds = torch.argmax(probs,dim=-1).sum().item()
        neg_preds = len(batch) - pos_preds

        neg_num += neg_preds
        pos_num += pos_preds

    pos_acc = round(pos_num / len(eval_data) * 100,4)
    neg_acc = round(neg_num / len(eval_data) * 100,4)
    if attr == "pos":
        return pos_acc,acc_list
    elif attr == "neg":
        return neg_acc,acc_list

@torch.no_grad()
def generation(args):
    num_sequence = args.num_sequence
    generate_length = args.generate_length
    collect_batch_size = args.batch_size
    agent = args.agent
    prompt_list = agent.reset(batch_size=collect_batch_size, generation=True)
    each_collect_epoch = num_sequence // collect_batch_size
    all_texts = []

    for p in tqdm(prompt_list, desc="Generating"):
        input_ids = agent.tokenizer.encode(p,return_tensors='pt').to(agent.device)
        init_batch_ids = input_ids.expand(collect_batch_size, -1)
        single_text_for_prompt = []
        for s in range(each_collect_epoch):
            batch_ids =  init_batch_ids
            for step in range(generate_length):
                action, _ = agent.get_action_log_prob(batch_ids, evaluate=True)
                batch_ids, _ = agent.step(batch_ids, action, _, evaluate=True)
            texts = agent.tokenizer.batch_decode(batch_ids)
            generate_text = [text.strip('\n').replace('\n', ' ') for text in texts]
            single_text_for_prompt.extend(generate_text)
        all_texts.extend(single_text_for_prompt)

    return all_texts

@torch.no_grad()
def compute_ppl(args, texts, eval_batch_size):
    agent = args.agent
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
        ppl_ = -(probs.sum(dim=-1, keepdim=True) / true_num)
        ppl = torch.exp(ppl_)
        ppl_list.append(ppl)
    ppl = torch.cat(ppl_list, dim=0).tolist()
    ppl = [x[0] for x in ppl]
    ave_ppl = round(sum(ppl)/len(ppl), 2)

    return ave_ppl,ppl


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