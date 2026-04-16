import json
from torch.utils.data import DataLoader,Dataset
from tqdm import trange, tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2ForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
import torch
class my_data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        left = item * 640
        right = (item+1)*640
        return self.data[left:right]

    def __len__(self):
        return len(self.data)//640

def data_process():
    tokens = []
    file_path = "/home/anke/DXZ/RCG/dataset/sentiment-imdb/imdb_5k_pos_tokenized.json"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():

            data = json.loads(line)
            tokens.extend(data.get("text"))
    long_tokens = []
    short_tokens = []
    data = my_data(tokens)
    dataloader = DataLoader(data, batch_size=128,shuffle=False,drop_last=True,collate_fn=lambda x: x)
    for batch in tqdm(dataloader):
        lt = [x[:512] for x in batch]
        long_tokens.extend(lt)
        st = [x[-128:] for x in batch]
        short_tokens.extend(st)
    long_file_path = "long_pos.jsonl"
    short_file_path = "short_pos.jsonl"

    f1 = open(long_file_path,"w")
    f2 = open(short_file_path,"w")
    for tl,ts in zip(long_tokens,short_tokens):
        ldict = {"text":tl,"label":1}
        sdict = {"text":ts,"label":1}
        f1.write(json.dumps(ldict,ensure_ascii=False)+"\n")
        f2.write(json.dumps(sdict,ensure_ascii=False)+"\n")
    f1.close()
    f2.close()




class eval_set(Dataset):
    def __init__(self,data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)

@torch.no_grad()
def gen_eval(long_lora_path,length):
    prompts = []
    device = torch.device("cuda")
    with open("/home/anke/DXZ/RCG/dataset/sentiment-imdb/prompt_sent.jsonl","r") as f:
        for line in f.readlines():
            obj = json.loads(line)
            prompts.append(obj['prompt'])




    batch = 8

    # long_lora_path = "/home/anke/DXZ/RCG/lora_train/sentiment/lora_ckpt/c_attn_lora/long_pos/epoch6_lora_acc0.672"
    short_lora_path = "/home/anke/DXZ/RCG/lora_train/sentiment/lora_ckpt/c_attn_lora/long_pos/epoch6_lora_acc0.672"




    model = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(device)
    model = PeftModel.from_pretrained(model,long_lora_path ).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    texts = []
    for prompt in tqdm(prompts,desc="generating"):
        input = [prompt for _ in range(batch)]
        input_ids = tokenizer(input,return_tensors="pt").input_ids.to(device)
        for gen_len in range(length):
            output = model(input_ids=input_ids).logits[:,-1,:].softmax(dim=-1)
            topk_probs, topk_ids = torch.topk(output,dim=-1,k=5)
            selected = topk_probs.multinomial(num_samples=1)
            next_token = topk_ids.gather(dim=-1,index=selected)
            input_ids = torch.cat([input_ids,next_token],dim=-1)
        texts.extend(tokenizer.batch_decode(input_ids))

    classifer = GPT2ForSequenceClassification.from_pretrained("/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment").to(device)
    eval_data = eval_set(texts)
    dataloader = DataLoader(eval_data, batch_size=batch, drop_last=False,shuffle=False)
    total_num= 0
    pos_num = 0
    for b in tqdm(dataloader,desc="eval"):
        total_num+=len(b)
        input = tokenizer(b,return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
        output = classifer(**input).logits.softmax(dim=-1)
        pos_num+=output.argmax(dim=-1).sum().item()
        pass

    return round(pos_num/total_num,2)





if __name__ == "__main__":
    long_lora_path = "/home/anke/DXZ/RCG/lora_train/sentiment/lora_ckpt/c_attn_lora/long_pos/epoch6_lora_acc0.672"
    short_lora_path = "/home/anke/DXZ/RCG/lora_train/sentiment/lora_ckpt/c_attn_lora/short_neg/epoch5_lora_acc0.82"
    long_acc= []
    short_acc = []
    long_test = [(long_lora_path,64),(long_lora_path,128),(long_lora_path,256),(long_lora_path,512)]
    short_test = [(short_lora_path,64),(short_lora_path,128),(short_lora_path,256),(short_lora_path,512)]

    for path, length in long_test:
        long_acc.append(gen_eval(path,length))
    for path, length in short_test:
        short_acc.append(gen_eval(path,length))
    print(f"long_test:{long_acc}\nshort_test:{short_acc}")



