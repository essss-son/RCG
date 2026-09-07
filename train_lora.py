import math
import os.path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from transformers import GPT2ForSequenceClassification, RobertaTokenizer, pipeline,RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.optim import AdamW
import torch.nn as nn
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

import random
class my_dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.text = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                self.text.append(obj['text'])
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return self.text[idx]

def padding_fuse(lst):
    length = []
    for text in lst:
        length.append(len(text))
    max_len = max(length)

    batch_ids = []
    attention_mask = []
    for text in lst:
        padding_len = max_len - len(text)
        padding = torch.full((padding_len,), 50256, dtype=torch.int)
        batch_ids.append(torch.cat([torch.tensor(text), padding], dim=0))

        attention_mask.append(torch.tensor([1] * len(text) + [0] * padding_len))

    batch_ids = torch.stack(batch_ids,dim=0)
    attention_mask = torch.stack(attention_mask,dim=0)
    start_tokens = torch.tensor(len(length) * [50256]).unsqueeze(1)
    batch_ids = torch.cat([start_tokens, batch_ids], dim=-1)
    start_mask = torch.tensor([1] * len(length)).unsqueeze(1)
    attention_mask = torch.cat([start_mask, attention_mask], dim=-1)
    return batch_ids, attention_mask

def main():
    model_path = "/home/anke/DXZ/models/gpt2-medium"

    task = "sentiment"

    lora_ckpt_path = "./lora_train/" + task + f"/lora_ckpt/{module_type}_lora/"
    os.makedirs(lora_ckpt_path, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    lr = 3e-5
    batch_size = 4
    train_epoch_for_each_dataset = 10
    writer= SummaryWriter(log_dir="./lora_train/" + task + f"/lora_logs/{module_type}_logs")
    dataset_path = [
        "/home/anke/DXZ/RCG/dataset/sentiment-imdb/imdb_5k_pos_tokenized.json",
        "/home/anke/DXZ/RCG/dataset/sentiment-imdb/imdb_5k_neg_tokenized.json"

        # "/home/anke/DXZ/RCG/dataset/topic-agnews/agnews_5k_world_tokenized.json",
        # "/home/anke/DXZ/RCG/dataset/topic-agnews/agnews_5k_sports_tokenized.json",
        # "/home/anke/DXZ/RCG/dataset/topic-agnews/agnews_5k_business_tokenized.json",
        # "/home/anke/DXZ/RCG/dataset/topic-agnews/agnews_5k_science_tokenized.json",

        # "/home/anke/DXZ/RCG/dataset/detoxification-jigsaw/jigsaw_5k_nontoxic_tokenized.json",
        # "/home/anke/DXZ/RCG/dataset/detoxification-jigsaw/jigsaw_5k_toxic_tokenized.json",

        # "/home/anke/DXZ/RCG/long_pos.jsonl",
        # "/home/anke/DXZ/RCG/short_pos.jsonl"

    ]
    writer_dict = {
        0:"pos",
        1:"neg",

        # 0: "world",
        # 1: "sports",
        # 2: "business",
        # 3: "science",
        #
        # 0:"nontoxic",
        # 1:"toxic",
        # 0: "long_pos",
        # 1: "short_neg",

    }
    accu_steps = 4
    for data_enum,dataset in enumerate(dataset_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # LoRA 的秩，越大控制力越强，但参数越多。8 或 16 通常够用。
            lora_alpha=16,  # 缩放系数，通常是 r 的 2-4 倍
            lora_dropout=0.1,
            target_modules=["c_attn"] if module_type == "c_attn" else ["c_fc", "c_proj"],
        )
        model = get_peft_model(model, peft_config).to(device)

        # total_num = 0
        # train_num = 0
        # for param in model.parameters():
        #     num = param.numel()
        #     total_num += num
        #     if param.requires_grad:
        #         train_num += num
        # print(
        #     f'Total number of parameters: {total_num}, Trainable parameters: {train_num}, Percentage: {round(train_num / total_num * 100, 2)}%')
        #
        # return

        my_set = my_dataset(dataset)
        total_l_steps = len(my_set) // batch_size * train_epoch_for_each_dataset
        dataloader = DataLoader(my_set, batch_size=batch_size, shuffle=True,collate_fn=padding_fuse,drop_last=True)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, math.floor(total_l_steps * 0.15), total_l_steps)

        loss_fn = nn.CrossEntropyLoss(ignore_index=50256)
        current_epoch = 0
        total_step = len(my_set) // batch_size
        optimizer.zero_grad()

        # evaluate(model, tokenizer)
        min_acc = -10
        for epoch in trange(train_epoch_for_each_dataset):
            model.train()
            current_epoch += 1
            # update_step = 0
            for step, batch in enumerate(dataloader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                target = input_ids[:,1:].contiguous()
                output = model(input_ids, attention_mask=attention_mask)
                shifted_logits = output.logits[:,:-1,:].contiguous()


                loss = loss_fn(shifted_logits.view(-1,shifted_logits.shape[-1]), target.view(-1))
                loss.backward()

                # if (step + 1) % accu_steps == 0:
                total_grad = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad += param.grad.data.norm(2) ** 2
                total_grad = total_grad ** 0.5
                writer.add_scalars('Grad', {writer_dict[data_enum]:total_grad.item()}, epoch * total_step + step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # update_step += 1
                writer.add_scalars('Lr', {writer_dict[data_enum]:scheduler.get_last_lr()[0]}, epoch * total_step + step)
                writer.add_scalars('Loss', {writer_dict[data_enum]:loss.item()}, epoch * total_step + step)
            model.eval()
            acc = evaluate(model,tokenizer,writer_dict[data_enum],task)
            # acc = evaluate(model,tokenizer,"pos",task,epoch)

            # save_dir = lora_ckpt_path + f"/{writer_dict[data_enum]}/epoch{epoch}_lora_toxic{acc}"
            save_dir = lora_ckpt_path + f"/{writer_dict[data_enum]}/epoch{epoch}_lora_acc{acc}"
            model.save_pretrained(save_dir)
        del model

def evaluate(model, tokenizer,attr,task):
    prompt_paths = {
        "sentiment":"/home/anke/DXZ/RCG/dataset/sentiment-imdb/prompt_sent.jsonl",
        "topic":"/home/anke/DXZ/RCG/dataset/topic-agnews/prompt_topic.jsonl",
        "detoxification":"/home/anke/DXZ/RCG/dataset/detoxification-jigsaw/prompt_detoxification.jsonl"
    }
    prompt_path = prompt_paths[task]
    lora_eval_file = "./lora_train/" + task + f"/Eval/{module_type}/{attr}/"

    os.makedirs(lora_eval_file,exist_ok=True)

    eval_batch = 8
    eval_sequence = 128
    gen_length = 50
    gen_epoch = eval_sequence // eval_batch
    prompt = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt.append(obj['prompt'])
    one_p = random.choice(prompt)
    prompts = [one_p for i in range(eval_batch)]

    all_texts = []
    with torch.no_grad():
        for i in range(gen_epoch):
            input_s = prompts
            try:
                encoded = tokenizer(input_s, return_tensors="pt").to(model.device)
                input_ids = encoded['input_ids']
            except Exception as e:
                print(e)
                breakpoint()
            for step in range(gen_length):
                output = model(input_ids).logits
                topk_probs,topk_indexs = output[:,-1,:].softmax(dim=-1).topk(k=200)
                idx = topk_probs.multinomial(num_samples=1)
                next_token = torch.gather(topk_indexs,dim=-1, index=idx)
                input_ids = torch.cat([input_ids, next_token],dim=-1)
            batch_text = tokenizer.batch_decode(input_ids,skip_special_tokens=True)
            all_texts.extend(batch_text)
        ave_acc, sent_score_list = eval_sent(all_texts,model.device,tokenizer,attr)
        # ave_acc = eval_topic(all_texts,model.device,attr)
        # ave_toxicity, toxicity = eval_toxicity(all_texts, device)
        ave_ppl, ppl_list = eval_ppl(all_texts,device, tokenizer)

    eval_file = lora_eval_file + f"/len{gen_length}_acc{round(ave_acc,2)}_ppl{round(ave_ppl,2)}.txt"
    # eval_file = lora_eval_file + f"/epoch{epoch}_len{gen_length}_toxicity{ave_toxicity}_ppl{round(ave_ppl,2)}.txt"
    with open(eval_file, "w", encoding="utf-8") as f:
        for line in all_texts:
            f.write(line.replace("\n"," ") + '\n')
    return ave_acc

    # with open(eval_file, "w", encoding="utf-8") as f:
    #     for line,t in zip(all_texts,toxicity):
    #         f.write(line.replace("\n"," ")+f"         toxicity:{t}" + '\n')
    # return ave_toxicity

class my_evalset(Dataset):
    def __init__(self,data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)


def eval_sent(texts,device,tokenizer,attr):
    model = GPT2ForSequenceClassification.from_pretrained("/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment").to(device)
    eval_batch = 16
    tokenizer.pad_token = tokenizer.eos_token
    eval_set = my_evalset(texts)
    loader = DataLoader(eval_set, batch_size=eval_batch,shuffle=False, drop_last=False)
    score_list = []
    pos_num = 0
    neg_num = 0

    for batch in loader:
        encoded = tokenizer(batch,return_tensors="pt",padding=True).to(device)
        score = model(**encoded).logits.softmax(dim=-1)
        if attr == "pos":
            score_list.extend(score[:,1].tolist())
        elif attr == "neg":
            score_list.extend(score[:,0].tolist())

        batch_pos = score.argmax(dim=-1).sum().item()
        pos_num += batch_pos
        neg_num += (eval_batch - batch_pos)

    pos_acc = round(pos_num/len(texts),3)
    neg_acc = round(neg_num/len(texts),3)
    if attr == "pos":
        return pos_acc, score_list
    else:
        return neg_acc, score_list


@torch.no_grad()
def eval_topic(texts,device,attr):
    attr_dict = {
        "world": 0,
        "sports": 1,
        "business": 2,
        "science": 3,
    }
    model = RobertaForSequenceClassification.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news")
    eval_batch = 16
    data = my_evalset(texts)
    loader = DataLoader(data, batch_size=eval_batch,shuffle=False, drop_last=False)
    correct = 0
    for batch in loader:
        encoded = tokenizer(batch,return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
        output = model(**encoded).logits.softmax(dim=-1)
        correct += (output.argmax(dim=-1)==attr_dict[attr]).sum(dim=-1).item()


    acc = round(correct / len(texts) * 100, 2)
    return acc

@torch.no_grad()
def eval_toxicity(texts,device):
    classifier = pipeline(
        "text-classification",
        model="/home/anke/DXZ/models/unbiased-toxic-roberta",
        return_all_scores=True,
        device=device,
        batch_size=32,
    )
    results = []
    for out in classifier(texts):
        score = [item['score'] for item in out if item['label'] == 'toxicity']
        results.append(score[0] * 100)
    ave_toxicity = round(sum(results) / len(results),2)

    return ave_toxicity, results

def eval_ppl(texts,device, tokenizer):
    model = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(device)
    eval_batch = 16
    my_set = my_evalset(texts)
    loader = DataLoader(my_set, batch_size=eval_batch,shuffle=False, drop_last=False)
    ppl_list = []
    total_ppl = 0.0
    for batch in loader:
        encoded = tokenizer(batch,return_tensors="pt",padding=True).to(device)
        target = encoded.input_ids[:,1:]
        output = model(**encoded).logits[:,:-1,:].log_softmax(dim=-1)
        p = output.gather(dim=-1,index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        ppl = torch.exp(-p.mean(dim=-1))
        ppl_list.extend(ppl.tolist())
        total_ppl += ppl.sum().item()
    ave_ppl = total_ppl / len(texts)
    return ave_ppl, ppl_list



if __name__ == '__main__':
    module_type = "c_attn"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()