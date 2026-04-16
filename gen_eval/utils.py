from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
def compute_sent_acc(args, texts, eval_batch_size, attr):
    agent = args.agent
    eval_data = my_eval_set(texts)
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size,shuffle=False,drop_last=False)
    pos_num = 0
    neg_num = 0
    acc_list = []
    agent.reward_tokenizer.pad_token = agent.reward_tokenizer.eos_token
    for batch in eval_loader:
        encoded = agent.reward_tokenizer(batch,return_tensors='pt',padding=True, truncation=True).to(agent.device)
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
def compute_toxicity(args,texts, eval_batch_size):
    agent = args.agent
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


@torch.no_grad()
def compute_topic_acc(args, texts, eval_batch_size, attr):
    agent = args.agent
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
        output = score_sentences_with_chunks(batch,model,tokenizer).tolist()
        label_list.extend(output)
        c_list = [1 if label == attr_dict[attr] else 0 for label in output]
        correct += sum(c_list)
    acc = round(correct / len(eval_set) * 100, 2)
    return acc, label_list



