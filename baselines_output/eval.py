import json

import torch
from transformers import GPT2ForSequenceClassification,GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer,RobertaTokenizer,RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class my_eval_set(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def score_sentences_with_chunks(texts, model, tokenizer,attr=None):
    attr_idx = None
    if attr is not None:
        attr_dict = {
            "world": 0,
            "sports": 1,
            "business": 2,
            "science": 3
        }
        attr_idx = attr_dict[attr]

    chunk_size = 64

    encode = tokenizer(texts, return_tensors="pt",truncation=True,max_length=512,padding=True).to(model.device)
    input_ids_list = encode.input_ids.split(chunk_size,dim=-1)
    attention_mask_list = encode.attention_mask.split(chunk_size,dim=-1)

    batch_score_list = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        score = model(input_ids=input_ids, attention_mask=attention_mask).logits.softmax(dim=-1)
        if attr_idx is not None:
            score = score[:, attr_idx:attr_idx + 1]
        else:
            pass
        batch_score_list.append(score * input_ids.shape[1] / chunk_size)

    final_score = torch.zeros_like(batch_score_list[0])
    for score in batch_score_list:
        final_score += score
    final_score = final_score / len(batch_score_list)

    if attr is not None:
        return final_score
    else:
        return final_score.argmax(dim=-1)


def score_toxic_with_chunks(texts, model, tokenizer):

    chunk_size = 64

    encode = tokenizer(texts, return_tensors="pt",truncation=True,max_length=512,padding=True).to(model.device)
    input_ids_list = encode.input_ids.split(chunk_size,dim=-1)
    attention_mask_list = encode.attention_mask.split(chunk_size,dim=-1)

    batch_score_list = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        score = model(input_ids=input_ids, attention_mask=attention_mask)[0].softmax(dim=-1)
        toxic_id = model.config.label2id['toxicity']
        toxic_score = score[:, toxic_id]

        batch_score_list.append(toxic_score * input_ids.shape[1] / chunk_size)

    final_score = torch.zeros_like(batch_score_list[0])
    for score in batch_score_list:
        final_score += score
    final_score = final_score / len(batch_score_list)


    return final_score





@torch.no_grad()
def compute_topic_acc(model,tokenizer, texts, eval_batch_size, attr):


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
        output = score_sentences_with_chunks(batch,model,tokenizer).tolist()
        label_list.extend(output)
        c_list = [1 if label == attr_dict[attr] else 0 for label in output]
        correct += sum(c_list)
    acc = round(correct / len(eval_set) * 100, 2)
    return acc, label_list


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


@torch.no_grad()
def compute_toxicity(model, tokenizer,device,texts, eval_batch_size):

    eval_set = my_eval_set(texts)
    loader = DataLoader(eval_set, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    total_toxic = []
    for batch in loader:
        input = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**input)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=-1)
        tox_id = model.config.label2id["toxicity"]
        tox_scores = probs[:, tox_id].tolist()  # shape: [B]
        total_toxic.extend(tox_scores)
    ave_toxic = round(sum(total_toxic) / len(total_toxic) * 100, 2)
    return ave_toxic, total_toxic



@torch.no_grad()
def compute_sent_acc(model, tk,device,texts, eval_batch_size, attr):

    eval_data = my_eval_set(texts)
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size,shuffle=False,drop_last=False)
    pos_num = 0
    neg_num = 0
    acc_list = []
    for batch in eval_loader:
        encoded = tk(batch,return_tensors='pt',padding=True).to(device)
        logits = model(**encoded).logits
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
def compute_ppl(model,tokenizer,device, texts, eval_batch_size):

    ppl_list = []
    my_data = my_eval_set(texts)
    my_loader = DataLoader(my_data, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    for batch in my_loader:
        encoded = tokenizer(batch,padding=True,return_tensors="pt").to(device)
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



def eval_sent(texts,attr):
    sent_model = GPT2ForSequenceClassification.from_pretrained(
        "/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment").to(device).eval()
    sent_tokenizer = ppl_tokenizer

    batch = 16

    acc, lst = compute_sent_acc(sent_model, sent_tokenizer, device,texts,batch,attr)
    ppl, _ = compute_ppl(ppl_model,ppl_tokenizer,device,texts,batch)
    d1,d2,d3 = compute_distinct(texts)

    return acc, ppl, d1,d2,d3



def eval_topic(texts,attr):
    topic_model = RobertaForSequenceClassification.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news").to(
        device).eval()
    topic_tokenizer = RobertaTokenizer.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news")

    batch = 32

    acc,lst = compute_topic_acc(topic_model, topic_tokenizer,texts,batch,attr)
    ppl, _ = compute_ppl(ppl_model, ppl_tokenizer, device, texts, batch)
    d1, d2, d3 = compute_distinct(texts)


    return acc, ppl, d1,d2,d3


def eval_toxic(texts):
    toxic_model = AutoModelForSequenceClassification.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta").to(
        device).eval()
    toxic_tokenizer = AutoTokenizer.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta")

    batch_size = 32
    dataset = my_eval_set(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    tox_list = []


    for batch in loader:
        tox =  score_toxic_with_chunks(batch,toxic_model,toxic_tokenizer)
        tox_list.extend(tox.tolist())
    tox = sum(tox_list)/len(tox_list) * 100
    ppl, _ = compute_ppl(ppl_model, ppl_tokenizer, device, texts, batch_size)
    d1, d2, d3 = compute_distinct(texts)

    return tox, ppl, d1,d2,d3


def read_air_data(path):
    text = []
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text.append(obj['text'])
    return text

def read_ctrl(path):
    text = []
    with open(path, 'r') as f:
        for line in f:

            text.append(line)
    return text


def read_tox_data(path):
    text = []
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text.append(obj['prompt']+obj['text']['0'])
    return text


if __name__ == '__main__':

    # pos_path64 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/pos/sentiment_length_64.jsonl"
    # pos_path128 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/pos/sentiment_length_128.jsonl"
    # pos_path256 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/pos/sentiment_length_256.jsonl"
    # pos_path512 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/pos/sentiment_length_512.jsonl"
    #
    # neg_path64 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/neg/sentiment_length_64.jsonl"
    # neg_path128 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/neg/sentiment_length_128.jsonl"
    # neg_path256 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/neg/sentiment_length_256.jsonl"
    # neg_path512 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/neg/sentiment_length_512.jsonl"
    #
    # topic_path64 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/topic/length_64.jsonl"
    # topic_path128 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/topic/length_128.jsonl"
    # topic_path256 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/topic/length_256.jsonl"
    # topic_path512 = "/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/topic/length_512.jsonl"






    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppl_model = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(device).eval()
    ppl_tokenizer = GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium")
    ppl_tokenizer.pad_token = ppl_tokenizer.eos_token






    # all_texts = read_tox_data("/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/detoxification/length_128.jsonl")
    all_texts = read_air_data("/home/anke/DXZ/freectrl/version_9/baselines_output/air-decoding/sentiment/pos/sentiment_length_512.jsonl")
    # all_texts = read_ctrl("/home/anke/DXZ/freectrl/version_9/baselines_output/Freectrl/topic/Business_Length_64.txt")
    # wd = all_texts[:320]
    # sd = all_texts[320:640]
    # bd = all_texts[640:960]
    # scd = all_texts[960:]

    acc, ppl, d1,d2,d3 = eval_sent(all_texts,"pos")
    # acc, ppl, d1,d2,d3 = eval_toxic(all_texts)
    # accw, pplw, d1w,d2w,d3w = eval_topic(wd,"world")
    # accs, ppls, d1s,d2s,d3s = eval_topic(sd,"sports")
    # acc, ppl, d1,d2,d3 = eval_topic(all_texts,"business")
    # accsc, pplsc, d1sc,d2sc,d3sc = eval_topic(scd,"science")
    # acc = round((accw+accs+accb+accsc)/4,2)
    # ppl = round((pplw+ppls+pplb+pplsc)/4,2)
    # d1 = round((d1w+d1s+d1b+d1sc)/4,2)
    # d2 = round((d2w+d2s+d2b+d2sc)/4,2)
    # d3 = round((d3w+d3s+d3b+d3sc)/4,2)

    print(f"acc: {acc}, ppl: {ppl}, d1: {d1}, d2: {d2}, d3: {d3}")













    pass