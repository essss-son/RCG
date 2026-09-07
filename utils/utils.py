from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch
import random
import json
import copy
import os
import torch.nn as nn
from train_lora import eval_sent
from model_utils.model import LayerController
from torch.utils.data import Dataset
import torch.nn.functional as F
class c_attn_lora_method:
    gpt_path = "/home/anke/DXZ/models/gpt2-medium"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = None
    _tokenizer = None

    @classmethod
    def get_gpt2(cls):
        if cls._model is None:
            cls._model = GPT2LMHeadModel.from_pretrained(cls.gpt_path)
            cls._tokenizer = GPT2Tokenizer.from_pretrained(cls.gpt_path)
        return copy.deepcopy(cls._model).to(cls.device), cls._tokenizer

    @classmethod
    def get_lora_params(cls,lora_path):
        """加载单个 LoRA"""
        # (保持原有逻辑不变)
        base_model = copy.deepcopy(cls._model if cls._model else GPT2LMHeadModel.from_pretrained(cls.gpt_path))
        lora_model = PeftModel.from_pretrained(base_model, lora_path)

        # 读取 adapter_config.json 获取 alpha 和 r
        config_path = os.path.join(lora_path, "adapter_config.json")
        lora_config = json.load(open(config_path, 'r'))
        scaling = lora_config['lora_alpha'] / lora_config['r']

        lora_param = cls.extract_param(lora_model)

        del base_model
        del lora_model
        torch.cuda.empty_cache()  # 清理显存
        return lora_param, scaling

    @classmethod
    def extract_param(cls, lora_model):
        """(保持原有逻辑不变)"""
        sd = lora_model.state_dict()
        extracted = {}
        for key, weight in sd.items():
            if "lora_A" in key or "lora_B" in key:
                parts = key.split('.')
                try:
                    h_index = parts.index('h')
                    layer_idx = int(parts[h_index + 1])
                except (ValueError, IndexError):
                    continue
                if layer_idx not in extracted:
                    extracted[layer_idx] = {}
                if "lora_A" in key:
                    extracted[layer_idx]['A'] = weight.detach().to(cls.device)
                elif "lora_B" in key:
                    extracted[layer_idx]['B'] = weight.detach().to(cls.device)
        return extracted

    #note 目前貌似没有用到
    @classmethod
    def set_layer_hooks(cls,base_model,  lora_param, scaling):
        """
            给 base_model 每层 c_attn 注册 hook，把对应层的 LoRA 输出加上去。
            pos_lora_param: 长度 = 层数，每层是 dict，包含 A, B, scaling
            """
        handles = []

        # 遍历每一层
        for layer_idx, layer in enumerate(base_model.transformer.h):
            lora_data = lora_param[layer_idx]  # dict: {'A': tensor, 'B': tensor, 'scaling': float}

            def make_hook(A, B, scaling):
                def lora_hook(module, input, output):
                    x = input[0]  # [B, T, hidden]
                    delta = (x @ A.t()) @ B.t() * scaling
                    return output + delta

                return lora_hook

            hook = make_hook(
                A=lora_data['A'],
                B=lora_data['B'],
                scaling=scaling
            )

            h = layer.attn.c_attn.register_forward_hook(hook)
            handles.append(h)

        return handles  # 返回 handle，方便之后移除

    @classmethod
    def remove_hooks(cls,hooks):
        for h in hooks:
            h.remove()


def inject_policy_hooks(base_model, lora_params_dict, scalings_dict, device, num_active_adapters=2):
    """
    lora_params_dict: { "pos": [param_layer_0, ...], "neg": [...], ... }
    scalings_dict: { "pos": 1.0, ... }
    """
    base_model.fusion_controllers = nn.ModuleList()
    handles = []

    # 假设所有 lora 都有相同的层数，取 GPT2 的层数
    num_layers = len(base_model.transformer.h)

    for i in range(num_layers):
        layer = base_model.transformer.h[i]
        c_attn = layer.attn.c_attn
        hidden_size = c_attn.weight.shape[0]
        qkv_output_dim = c_attn.weight.shape[1]

        # MODIFIED: 收集当前层的所有 LoRA 参数
        current_layer_params = {}
        for name, params_list in lora_params_dict.items():
            if params_list[i] is not None:
                current_layer_params[name] = params_list[i]

        # 实例化 Controller
        controller = LayerController(
            layer_idx=i,
            hidden_size=hidden_size,
            qkv_output_dim=qkv_output_dim,
            all_adapters_params=current_layer_params,  # 传入字典
            all_scalings=scalings_dict,
            num_active_adapters=num_active_adapters
        ).to(device)

        base_model.fusion_controllers.append(controller)
        h = c_attn.register_forward_hook(controller.hook_fn)
        handles.append(h)

    print(f"Successfully injected Policy into {len(handles)} layers with {len(lora_params_dict)} adapters each.")
    return handles

def get_injected_model(lora_paths_dict, device, num_active_adapters=2):
    """
    lora_paths_dict: { "pos": "/path/to/pos", "world": "/path/to/world", ... }
    num_active_adapters: 训练/推理时打算同时开启几个 adapter (决定 Policy 输入维度)
    """
    lora_params_dict = {}
    scalings_dict = {}

    # 1. 加载所有 LoRA 参数
    print("Loading LoRA adapters...")
    for name, path in lora_paths_dict.items():
        print(f"  - Loading {name} from {path}")
        params, scaling = c_attn_lora_method.get_lora_params(path)
        lora_params_dict[name] = params
        scalings_dict[name] = scaling

    # 2. 加载 Base Model
    model, _ = c_attn_lora_method.get_gpt2()

    # 3. 注入
    inject_policy_hooks(model, lora_params_dict, scalings_dict, device, num_active_adapters)

    # 4. 设置梯度
    for name, param in model.named_parameters():
        if "fusion_controllers" in name and "policy" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model




class reward_data(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


class reward_dataset(Dataset):
    def __init__(self, data):
        self.data = data



    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


@torch.no_grad()
def reward_fn(next_state,args,reward_model,reward_tokenizer,ref_model,ref_tokenizer):
    attr_score = None
    reward_scaling = args.reward_scaling
    base_probs_scaling = args.base_probs_scaling
    repeat_scaling = args.repeat_scaling

    idx = next_state[:, -1:]
    repeat_score = calculate_ngram_repetition(next_state, 2) * repeat_scaling
    base_logits = ref_model(next_state[:, :-1]).logits[:, -1, :].log_softmax(dim=-1)
    base_score = base_logits.gather(dim=-1, index=idx) * base_probs_scaling

    if args.task == "sentiment":
        if args.attr == "pos":
            attr_score = reward_model(next_state).logits.softmax(dim=-1)[:, 1:] * reward_scaling
            # final_score = attr_score + base_score - repeat_score
        elif args.attr == "neg":
            attr_score = reward_model(next_state).logits.softmax(dim=-1)[:,:1] * reward_scaling
            # final_score = attr_score + base_score - repeat_score

    elif args.task == "topic":
        attr_dict = {
            "world":0,
            "sports":1,
            "business":2,
            "science":3
        }
        attr_idx = attr_dict[args.attr]
        texts = ref_tokenizer.batch_decode(next_state, skip_special_tokens=True)
        encoded = reward_tokenizer(texts,return_tensors="pt",padding=True,max_length=512,truncation=True).to(args.device)
        new_next_state = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        attr_score = reward_model(input_ids=new_next_state,attention_mask=attention_mask).logits.softmax(dim=-1)[:, attr_idx:attr_idx + 1] * reward_scaling
        # final_score = attr_score + base_score - repeat_score

    elif args.task == "detoxification":
        texts = ref_tokenizer.batch_decode(next_state, skip_special_tokens=True)
        input = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(args.device)
        outputs = reward_model(**input)[0].softmax(dim=-1)
        tox_id = reward_model.config.label2id["toxicity"]
        attr_score = (1 - outputs[:, tox_id].unsqueeze(1)) * reward_scaling
        # final_score = attr_score + base_score - repeat_score

    if attr_score is None:
        raise ValueError("No final score found")



    # final_score = attr_score + base_score - repeat_score

    # if final_score is None:
    #     raise ValueError("No final score found")
    return torch.tanh(attr_score) , base_score, repeat_score


def calculate_ngram_repetition(input_ids: torch.Tensor, n: int) -> torch.Tensor:
    """
    计算 batch 中每个序列的 n-gram 重复率。

    参数:
        input_ids: 形状为 (batch_size, seq_len) 的 int tensor
        n: n-gram 的大小 (例如 2, 3, 4)

    返回:
        形状为 (batch_size, 1) 的 float tensor，范围在 [0.0, 1.0] 之间。
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # 如果序列长度小于 n，无法构成 n-gram，直接返回全 0
    if seq_len < n:
        return torch.zeros((batch_size, 1), device=device)

    # 将 tensor 转为 list 处理，因为 set 操作在 CPU 上对 python tuple 更方便高效
    # 也可以使用 unfold + unique，但在处理 batch 维度时，纯 PyTorch 操作较繁琐且显存占用大
    batch_list = input_ids.tolist()
    rep_rates = []

    for seq in batch_list:
        # 生成当前序列的所有 n-gram
        # zip(seq, seq[1:], seq[2:]...) 的方式是一种高效生成 n-gram 的 python trick
        ngrams = list(zip(*[seq[i:] for i in range(n)]))

        total_ngrams = len(ngrams)

        if total_ngrams == 0:
            rep_rates.append(0.0)
            continue

        unique_ngrams = set(ngrams)
        num_unique = len(unique_ngrams)

        # 计算重复率: 1 - (唯一数量 / 总数量)
        # 举例: A A A (2-gram: AA, AA). 总2, 唯一1. Rep = 1 - 0.5 = 0.5
        rate = 1.0 - (num_unique / total_ngrams)
        rep_rates.append(rate)

    # 转回 Tensor 并调整形状为 [batch_size, 1]
    return torch.tensor(rep_rates, device=device, dtype=torch.float32).unsqueeze(1)

class MyDataset(Dataset):
    def __init__(self, batch_states, batch_actions,batch_log_prob,batch_adv,batch_return):
        self.ids_state = []
        for step in batch_states:
            for sample in step:
                self.ids_state.append(sample)
        self.action_state = []
        for step in batch_actions:
            for sample in step:
                self.action_state.append(sample)
        self.log_prob_state = []
        for step in batch_log_prob:
            for sample in step:
                self.log_prob_state.append(sample)
        self.adv_state = []
        for step in batch_adv:
            for sample in step:
                self.adv_state.append(sample)
        self.return_state = []
        for step in batch_return:
            for sample in step:
                self.return_state.append(sample)

        assert len(self.ids_state) == len(self.action_state) == len(self.log_prob_state) == len(self.adv_state) == len(self.return_state), "MyDataset error"

    def __getitem__(self, item):
        return self.ids_state[item], self.action_state[item],self.log_prob_state[item],self.adv_state[item],self.return_state[item]

    def __len__(self):
        return len(self.ids_state)


def ppo_collate(lst):
    pad_token_id = 50256
    ids = []
    actions = []
    log_probs = []
    adv_state = []
    ret_state = []
    for sample in lst:
        ids.append(sample[0])
        actions.append(sample[1])
        log_probs.append(sample[2])
        adv_state.append(sample[3])
        ret_state.append(sample[4])
    ids_length = []
    for id in ids:
        ids_length.append(len(id))
    max_len = max(ids_length)

    new_ids = []
    attention_masks = []
    for id in ids:
        padding_length = max_len - len(id)
        no_padding = torch.ones_like(id).to(torch.int)
        padding = torch.zeros(padding_length, device=id.device).to(torch.int)
        ids_pad = torch.full((padding_length,), pad_token_id).to(id.device)
        new_ids.append(torch.cat([id, ids_pad], dim=-1))
        attention_masks.append(torch.cat([no_padding, padding], dim=-1))
    new_ids = torch.stack(new_ids, dim=0)
    new_actions = torch.stack(actions, dim=0)  # size batch,1
    new_adv = torch.stack(adv_state, dim=0)
    new_return = torch.stack(ret_state, dim=0)
    new_log_probs = torch.stack(log_probs, dim=0)
    new_mask = torch.stack(attention_masks, dim=0)

    return new_ids.detach(), new_actions.detach(), new_log_probs.detach(), new_adv.detach(), new_return.detach(), new_mask.detach()


@torch.no_grad()
def test_c_attn():
    neg_lora_path = "/home/anke/DXZ/freectrl/version_9/lora_train/lora_ckpt/c_attn_lora/neg_lora"
    pos_lora_path = "/home/anke/DXZ/freectrl/version_9/lora_train/lora_ckpt/c_attn_lora/pos_lora"
    base_model,base_tokenizer =c_attn_lora_method.get_gpt2()
    lora_param, scaling = c_attn_lora_method.get_lora_params(neg_lora_path)
    hooks = c_attn_lora_method.set_layer_hooks(base_model, lora_param, scaling)
    evaluate(base_model, base_tokenizer,"neg")
    c_attn_lora_method.remove_hooks(hooks)


def evaluate(model, tokenizer,attr):
    prompt_path = "/home/anke/DXZ/freectrl/version_9/data/prompt_sent.jsonl"

    eval_batch = 8
    eval_sequence = 256
    gen_length = 50
    gen_epoch = eval_sequence // eval_batch
    prompt = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt.append(obj['prompt'])
    prompts = [random.choice(prompt) for i in range(eval_batch)]
    all_texts = []
    for i in range(gen_epoch):
        input_s = prompts
        encoded = tokenizer(input_s, return_tensors="pt").to(model.device)
        input_ids = encoded['input_ids']
        for step in range(gen_length):
            output = model(input_ids).logits
            topk_probs,topk_indexs = output[:,-1,:].softmax(dim=-1).topk(k=200)
            idx = topk_probs.multinomial(num_samples=1)
            next_token = torch.gather(topk_indexs,dim=-1, index=idx)
            input_ids = torch.cat([input_ids, next_token],dim=-1)
        batch_text = tokenizer.batch_decode(input_ids,skip_special_tokens=True)
        all_texts.extend(batch_text)
    ave_acc, sent_score_list = eval_sent(all_texts,model.device,tokenizer,attr)
    for text in all_texts[:10]:
        print(text)
    print(ave_acc)


def col_softmax(x: torch.Tensor, col_idx: int):
    assert x.dim() == 2, "x must be a 2D tensor (batch_size, num_cols)"
    assert 0 <= col_idx < x.size(1), "col_idx out of range"

    selected = x[:, col_idx]  # (batch_size,)
    others_sum = x.sum(dim=1) - selected  # (batch_size,)

    scores = torch.stack([selected, others_sum], dim=1)
    return F.softmax(scores, dim=1)

if __name__ == '__main__':
    lora_path_dict = {
        "pos":"/home/anke/DXZ/freectrl/version_9/lora_train/sentiment/lora_ckpt/c_attn_lora/pos_lora_acc0.703",
        "neg":"/home/anke/DXZ/freectrl/version_9/lora_train/sentiment/lora_ckpt/c_attn_lora/neg_lora_acc0.75",
        "world":"/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/world/lora_acc82.81",
        "sports":"/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/sports/lora_acc87.5",
        "business":"/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/business/lora_acc64.06",
        "science":"/home/anke/DXZ/freectrl/version_9/lora_train/topic/lora_ckpt/c_attn_lora/science/lora_acc92.19",
        "toxic":"/home/anke/DXZ/freectrl/version_9/lora_train/detoxification/lora_ckpt/c_attn_lora/toxic/epoch14_lora_toxic91.84",
        "nontoxic":"/home/anke/DXZ/freectrl/version_9/lora_train/detoxification/lora_ckpt/c_attn_lora/nontoxic/epoch14_lora_toxic22.93",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attr = "pos"
    model = None
    if attr == "pos":
        model = get_injected_model(lora_path_dict,device,num_active_adapters=2)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['pos', 'neg'])
    elif attr == "neg":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=2)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['neg', 'pos'])
    elif attr == "world":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=1)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['world'])
    elif attr == "sports":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=1)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['sports'])
    elif attr == "business":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=1)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['business'])
    elif attr == "science":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=1)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['science'])
    elif attr == "toxic":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=2)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['toxic', 'nontoxic'])
    elif attr == "nontoxic":
        model = get_injected_model(lora_path_dict, device, num_active_adapters=2)
        for controller in model.fusion_controllers:
            controller.set_active_adapters(['nontoxic', 'toxic'])
    if model is None:
        raise ValueError("模型不能是none")
    for name, prams in model.named_parameters():
        print(name,prams.requires_grad)





    pass