
import torch.nn as nn
import json
from torch.optim import AdamW
import torch
import random
from torch.utils.data import DataLoader,Dataset
from itertools import chain
from transformers import GPT2ForSequenceClassification,GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer,RobertaTokenizer,RobertaForSequenceClassification
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
#note IDE
# from version_9.model_utils.model_topic import RobertaForPreTraining
# from version_9.model_utils.model import Critic

#note 跑脚本
# from model_utils.model import Critic


from peft import LoraConfig, get_peft_model,TaskType
import os


class FusionPolicy(nn.Module):
    def __init__(self, hidden_size, qkv_output_dim, num_active_adapters):
        """
        num_active_adapters: 策略网络预期接收的 adapter 数量。
                             例如：如果是 'Pos' + 'Neg' 对比，这里填 2。
                             如果是 单一 Topic，这里填 1。
        """
        super().__init__()

        # 动态计算输入维度
        # Input = Base(1) + Adapters(N)
        self.input_multiplier = 1 + num_active_adapters
        input_dim = qkv_output_dim * self.input_multiplier

        bottle_dim = 64

        self.net = nn.Sequential(
            nn.Linear(input_dim, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, qkv_output_dim)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, 1),
        )
        self.final_gate = nn.Sigmoid()
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)
        nn.init.constant_(self.gate_net[-1].bias, 0.0)

    def forward(self, base_out, adapter_outputs):
        """
        adapter_outputs: List[Tensor], 长度必须等于 self.input_multiplier - 1
        """
        # 校验输入数量
        if len(adapter_outputs) != (self.input_multiplier - 1):
            raise ValueError(f"Policy expected {self.input_multiplier - 1} adapters, got {len(adapter_outputs)}")

        # 1. 拼接:[Batch, Seq, Dim * (1+N)]
        cat_input = torch.cat([base_out] + adapter_outputs, dim=-1)

        # 2. 计算残差基准
        if len(adapter_outputs) == 2:
            residual_base = adapter_outputs[0] - adapter_outputs[1]
        elif len(adapter_outputs) == 1:
            residual_base = adapter_outputs[0]
        else:
            raise ValueError("残差输入出错，请检查 num_active_adapters")

        # 3. 计算最终方向和权重
        direction = self.net(cat_input) + residual_base
        alpha = self.final_gate(self.gate_net(cat_input) * 6)

        return direction * alpha

class LayerController(nn.Module):
    def __init__(self, layer_idx, hidden_size, qkv_output_dim,
                 all_adapters_params, all_scalings, num_active_adapters=2):
        super().__init__()
        self.layer_idx = layer_idx

        self.lora_A_bank = nn.ParameterDict()
        self.lora_B_bank = nn.ParameterDict()
        self.scalings = {}

        for name, params in all_adapters_params.items():
            self.lora_A_bank[name] = nn.Parameter(params['A'], requires_grad=False)
            self.lora_B_bank[name] = nn.Parameter(params['B'], requires_grad=False)
            self.scalings[name] = all_scalings[name]

        self.active_adapter_names = []
        self.policy = FusionPolicy(hidden_size, qkv_output_dim, num_active_adapters)

    def set_active_adapters(self, adapter_names):
        for name in adapter_names:
            if name not in self.lora_A_bank:
                raise ValueError(f"Adapter '{name}' not found in layer {self.layer_idx}")
        self.active_adapter_names = adapter_names

    def compute_lora(self, x, name):
        A = self.lora_A_bank[name]
        B = self.lora_B_bank[name]
        scaling = self.scalings[name]
        x = x.to(A.dtype)
        return (x @ A.t()) @ B.t() * scaling

    def hook_fn(self, module, input, output):
        # input[0] 通常是 hidden_states
        x = input[0]

        # =================【核心修复】=================
        # Hugging Face GPT2Attention 返回的 output 是一个 tuple: (attn_output, present, ...)
        # 我们只能对 tuple 的第一个元素（真正的输出 Tensor）进行操作
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            base_out = output[0]
        else:
            base_out = output
        # ==============================================

        adapter_outputs = []
        for name in self.active_adapter_names:
            out = self.compute_lora(x, name)
            adapter_outputs.append(out)

        # 传入 Policy 网络的必须是纯 Tensor
        fusion_delta = self.policy(base_out, adapter_outputs)

        # 计算注入后的新输出 Tensor
        new_base_out = base_out + fusion_delta

        # =================【返回重组】=================
        # 如果原始模型返回的是 tuple，必须将其重新打包返回
        if is_tuple:
            return (new_base_out,) + output[1:]
        else:
            return new_base_out
        # ==============================================



class Critic(nn.Module):
    def __init__(self, base_model,args):
        super().__init__()
        self.base_model = base_model
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.critic_lora_r,
            lora_alpha=args.critic_lora_alpha,
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )
        self.transformer = get_peft_model(self.base_model, peft_config)

        self.critic_head = nn.Linear(self.base_model.config.hidden_size, 1)

        nn.init.normal_(self.critic_head.weight, std=0.01)
        nn.init.constant_(self.critic_head.bias, 0.0)
    def forward(self, states,attention_mask=None):
        x = self.transformer(states, attention_mask=attention_mask)[0]
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1  # Shape: [batch_size]
            length = torch.arange(len(last_token_indices))
            x = x[length, last_token_indices, :]
        else:
            x = x[:, -1, :]
        x = self.critic_head(x)
        return x

    def save_critic(self,save_path):
        os.makedirs(save_path, exist_ok=True)
        self.transformer.save_pretrained(save_path)
        torch.save(self.critic_head.state_dict(), os.path.join(save_path, "critic_head.pt"))


class my_eval_set(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


class Agent:
    def __init__(self,args):
        self.args = args
        self.device = args.device

        self.gpt2 = self.get_train_model(args.attr)

        self.policy = self.gpt2.transformer
        self.lm_head = self.gpt2.lm_head

        # self.ref_gpt2, self.tokenizer = c_attn_lora_method.get_gpt2()
        self.ref_gpt2, self.tokenizer = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(self.device),GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_model, self.reward_tokenizer = self.get_reward_model_tokenizer(args.task)
        # for name, p in self.gpt2.named_parameters():
        #     print(name, p.requires_grad)
        self.actor_optimizer = AdamW(filter(lambda p:p.requires_grad,self.gpt2.parameters()), lr=args.actor_lr,weight_decay=5e-4)

        self.critic_base = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(self.device)
        self.critic = Critic(self.critic_base.transformer,args).to(self.device)
        # for name, p in self.critic.named_parameters():
        #     print(name, p.requires_grad)
        self.critic_optimizer = AdamW(filter(lambda p:p.requires_grad,self.critic.parameters()), lr=args.critic_lr)
        # self.decay_rate = 0.15
        # self.scheduler = get_linear_schedule_with_warmup(self.critic_optimizer,self.decay_rate * args.total_train_epoch * args.ppo_update_epoch, args.total_train_epoch * args.ppo_update_epoch)
        self.ref_gpt2 = GPT2LMHeadModel.from_pretrained("/home/anke/DXZ/models/gpt2-medium").to(self.device).eval()
        self.freeze(self.ref_gpt2)
        self.freeze(self.reward_model)

        self.noise_decay = 1

        trainable = 0
        total = 0
        # for name, p in self.gpt2.named_parameters():
        #     total += p.numel()
        #     if p.requires_grad:
        #         print(name, p.numel())
        #         trainable += p.numel()
        #
        # print(f"Trainable params:{trainable}, Total params: {total}, Percent{trainable/total*100:.2f}%")
        # print("Approx size (MB):", trainable * 4 / 1024 / 1024)

    def get_reward_model_tokenizer(self,task):
        model = None
        tokenizer = None
        if task == "sentiment":
            model = GPT2ForSequenceClassification.from_pretrained("/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment").to(self.device).eval()
            tokenizer = GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment")
            tokenizer.pad_token = tokenizer.eos_token
        elif task == "topic":
            model = RobertaForSequenceClassification.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news").to(self.device).eval()
            tokenizer = RobertaTokenizer.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news")
        elif task == "detoxification":
            model = AutoModelForSequenceClassification.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta").to(self.device).eval()
            tokenizer = AutoTokenizer.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta")
        if model is None or tokenizer is None:
            raise Exception("No model or tokenizer found")
        return model, tokenizer

    def get_train_model(self,attr):
        model = None
        # if attr == "pos":
            # model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=2)
            # for controller in model.fusion_controllers:
            #     controller.set_active_adapters(['pos', 'neg'])
        if attr == "pos":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['pos'])
        elif attr == "neg":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=2)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['neg', 'pos'])
        elif attr == "world":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['world'])
        elif attr == "sports":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['sports'])
        elif attr == "business":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['business'])
        elif attr == "science":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['science'])
        # elif attr == "nontoxic":
        #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=2)
        #     for controller in model.fusion_controllers:
        #         controller.set_active_adapters(['nontoxic', 'toxic'])
        elif attr == "nontoxic":
            model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            for controller in model.fusion_controllers:
                controller.set_active_adapters(['nontoxic'])
        if model is None:
            raise ValueError("模型不能是none")
        return model
    def freeze(self,model):
        for p in model.parameters():
            p.requires_grad = False

    def save_trainable_params(self, model,save_path):
        trainable_params = {
            name: param.detach().cpu()
            for name, param in model.named_parameters() if param.requires_grad
        }
        torch.save(trainable_params, save_path)

    def save(self,save_path):
        self.save_trainable_params(self.gpt2,save_path + 'actor_weights.pt')
        self.save_trainable_params(self.critic,save_path + 'critic_weights.pt')

    def load_trainable_params(self, model, save_path):
        state = torch.load(save_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # print("Missing keys:", missing)
        for name in missing:
            if "policy" in name:
                raise ValueError("model loading error, missing parameters")
        if unexpected != []:
            raise ValueError("model loading error, unexpected parameters")
        # print("Unexpected keys:", unexpected)

    def load(self,save_path):
        self.load_trainable_params(self.gpt2, save_path +'/actor_weights.pt')
        self.load_trainable_params(self.critic, save_path + '/critic_weights.pt')



    def reset(self, batch_size, generation=False):
        prompt_paths = {
            "sentiment":"/home/anke/DXZ/RCG/version_9/dataset/sentiment-imdb/prompt_sent.jsonl",
            "topic":"/home/anke/DXZ/freectrl/version_9/dataset/topic-agnews/prompt_topic.jsonl",
            "detoxification":"/home/anke/DXZ/freectrl/version_9/dataset/detoxification-jigsaw/prompt_detoxification.jsonl",
        }

        prompt_file = prompt_paths[self.args.task]
        prompt_list = []
        with open(prompt_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                prompt_list.append(obj['prompt'])
        if generation:
            return prompt_list

        single_prompt = random.choice(prompt_list)
        prompt = [single_prompt for i in range(batch_size)]
        state = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.input_ids = state["input_ids"]
        return state

    @torch.no_grad()
    def get_action_log_prob(self, states, evaluate=False):
        TOPK = self.args.TOPK
        TEMPERATURE = self.args.TEMPERATURE

        hidden_states = self.policy(states)[0][:,-1,:]
        logits = self.lm_head(hidden_states)

        if evaluate:
            e_logits = (logits/TEMPERATURE).softmax(dim=-1)
            topk_probs, topk_indices = torch.topk(e_logits,k=TOPK)
            next_idx = torch.multinomial(topk_probs,1)
            next_token = torch.gather(topk_indices,dim=-1,index=next_idx)
            return next_token, torch.zeros_like(next_token)

        value = self.critic(states)
        dist = Categorical(logits=logits/TEMPERATURE)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ref_logits = self.ref_gpt2(states).logits[:, -1, :]
        # ref_logits = self.ref_gpt2(self.input_ids).logits[:, -1, :]
        # ref_dist = Categorical(logits=ref_logits)
        # ref_log_probs = ref_dist.log_prob(action)
        # kl_d = log_prob - ref_log_probs



        if self.args.NOISE_DECAY != 0:

            decay = 0.9995
            noise = torch.randn((states.shape[0],1),device=states.device) + 1
            noise = noise*(decay ** self.noise_decay)
            self.noise_decay += 1
        else:
            noise = torch.randn((states.shape[0], 1), device=states.device) + 1



        return action.unsqueeze(1).detach(), log_prob.unsqueeze(1).detach(), value.detach(), noise.detach()

    def get_log_prob(self,states, action, attention_mask):
        TEMPERATURE = self.args.TEMPERATURE

        value = self.critic(states,attention_mask=attention_mask)
        hidden_states = self.policy(states, attention_mask=attention_mask)[0]

        last_token_indices = attention_mask.sum(dim=1) - 1  # Shape: [batch_size]
        length = torch.arange(len(last_token_indices))
        hidden_states = hidden_states[length,last_token_indices,:]

        logits = self.lm_head(hidden_states)
        dist = Categorical(logits=logits/TEMPERATURE)
        if action.dim() > 1:
            action = action.squeeze(-1)
        log_prob = dist.log_prob(action)
        return log_prob.unsqueeze(1), value




    def step(self,batch_ids, action, noise, collect_steps=None,evaluate=False):

        next_state = torch.cat([batch_ids,action], dim=-1)
        if evaluate:
            return next_state, torch.zeros_like(action)
        NOISE_SCALING = self.args.NOISE_SCALING

        attr_score , base_score, repeat_score = reward_fn(next_state,self.args,self.reward_model,self.reward_tokenizer,self.ref_gpt2, self.tokenizer)
        total_reward = attr_score + base_score - repeat_score

        noise = noise * NOISE_SCALING
        reward = total_reward + noise
        if collect_steps is not None:
            self.args.writer.add_scalar("Reward in trj",reward[0].item(),collect_steps)
            reward_dict = {
                "Attr reward":attr_score.mean().item(),
                "Noise":noise.mean().item(),
                "Base reward":base_score.mean().item(),
                "Repeat penalty":repeat_score.mean().item(),
                "Total reward":reward.mean().item()
            }
            self.args.writer.add_scalars("Reward_monitor",reward_dict,collect_steps)

        return next_state, reward



    def ppo_update(self, batch_states, batch_actions,batch_log_prob,batch_adv,batch_return,write_steps):
        ppo_update_epoch = self.args.ppo_update_epoch
        batch_size = self.args.update_batch_size
        my_data = MyDataset(batch_states, batch_actions,batch_log_prob,batch_adv,batch_return)
        dataloader = DataLoader(my_data,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=ppo_collate)

        for update_epoch in range(ppo_update_epoch):
            for i, batch in enumerate(dataloader):
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                batch_ids = batch[0]
                batch_action = batch[1]
                batch_log_prob = batch[2]
                batch_adv = batch[3]
                batch_return = batch[4]
                batch_mask = batch[5]

                new_log_prob, value = self.get_log_prob(batch_ids,batch_action,batch_mask)
                ratio = torch.exp(new_log_prob - batch_log_prob)
                self.args.writer.add_scalar('Ratio', ratio.mean(), write_steps)

                clip_ratio = torch.clamp(ratio,min=0.8,max=1.2)
                policy_loss = -torch.min(batch_adv * clip_ratio,ratio * batch_adv).mean()
                policy_loss.backward()
                # critic_dict = {
                #     "return":batch_return.mean().item(),
                #     "value":value.mean().item(),
                # }
                # self.args.writer.add_scalars('Critic monitor', critic_dict, write_steps)

                critic_loss = 0.5 * F.mse_loss(value,batch_return)
                critic_loss.backward()
                loss_dict = {
                    'policy_loss': policy_loss.item(),
                    'critic_loss': critic_loss.item(),
                }
                self.args.writer.add_scalars('Loss',loss_dict,write_steps)

                #note 统计lora和critic参数分布

                # note 考虑增加entrophy loss，如果action很快收敛的话


                policy_grad = 0.0
                for param in chain(self.gpt2.parameters(),self.critic.parameters()):
                    if param.grad is not None:
                        policy_grad += param.grad.data.norm(2) ** 2
                policy_grad = policy_grad ** 0.5
                self.args.writer.add_scalar('Grad', policy_grad, write_steps)
                torch.nn.utils.clip_grad_norm_(chain(self.gpt2.parameters(),self.critic.parameters()), max_norm=2.0)

                write_steps += 1

                self.actor_optimizer.step()
                self.critic_optimizer.step()
            # self.scheduler.step()
        return write_steps


from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch
import random
import json
import copy
import os
import torch.nn as nn
from train_lora import eval_sent

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
            cls._model = AutoModelForCausalLM.from_pretrained("/home/anke/DXZ/models/distilgpt2")
            cls._tokenizer = AutoTokenizer.from_pretrained("/home/anke/DXZ/models/distilgpt2")
        return copy.deepcopy(cls._model).to(cls.device), cls._tokenizer

    @classmethod
    def get_lora_params(cls, lora_path):
        """加载单个 LoRA (适配 DistilGPT2)"""
        # 初始化 base_model 为 DistilGPT2
        from transformers import AutoModelForCausalLM
        base_model = copy.deepcopy(cls._model if cls._model else AutoModelForCausalLM.from_pretrained("/home/anke/DXZ/models/distilgpt2"))

        # 加载 LoRA
        lora_model = PeftModel.from_pretrained(base_model, lora_path)

        # 读取 adapter_config.json 获取 alpha / r
        config_path = os.path.join(lora_path, "adapter_config.json")
        with open(config_path, 'r') as f:
            lora_config = json.load(f)
        scaling = lora_config['lora_alpha'] / lora_config['r']

        # 提取参数
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

    # DistilGPT2 层数
    num_layers = len(base_model.transformer.h)

    for i in range(num_layers):
        layer = base_model.transformer.h[i]

        attention = getattr(layer, "attn", getattr(layer, "attention", None))
        if attention is None:
            raise ValueError(f"No attention found in layer {i}")

        # 获取 hidden_size
        if hasattr(attention, "embed_dim"):
            hidden_size = attention.embed_dim
        else:
            raise ValueError(f"Cannot infer hidden size for layer {i}")

        # =================【关键修复】=================
        # LoRA 是作用在 c_attn (QKV) 上的，必须把 Hook 挂在 c_attn 上
        target_module = getattr(attention, "c_attn", None)
        if target_module is None:
            raise ValueError(f"c_attn module not found in layer {i}")

        # c_attn 一次性输出 Q, K, V，因此输出维度是 3 倍的 hidden_size
        qkv_output_dim = 3 * hidden_size
        # ==============================================

        # 收集当前层的所有 LoRA 参数
        current_layer_params = {}
        for name, params_list in lora_params_dict.items():
            if params_list[i] is not None:
                current_layer_params[name] = params_list[i]

        # 实例化 Controller
        controller = LayerController(
            layer_idx=i,
            hidden_size=hidden_size,
            qkv_output_dim=qkv_output_dim,
            all_adapters_params=current_layer_params,
            all_scalings=scalings_dict,
            num_active_adapters=num_active_adapters
        ).to(device)

        base_model.fusion_controllers.append(controller)

        # 将 hook 挂在 c_attn 的 forward 上
        h = target_module.register_forward_hook(controller.hook_fn)
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
    return attr_score , base_score, repeat_score


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


# @torch.no_grad()
# def test_c_attn():
#     neg_lora_path = "/home/anke/DXZ/freectrl/version_9/lora_train/lora_ckpt/c_attn_lora/neg_lora"
#     pos_lora_path = "/home/anke/DXZ/freectrl/version_9/lora_train/lora_ckpt/c_attn_lora/pos_lora"
#     base_model,base_tokenizer =c_attn_lora_method.get_gpt2()
#     lora_param, scaling = c_attn_lora_method.get_lora_params(neg_lora_path)
#     hooks = c_attn_lora_method.set_layer_hooks(base_model, lora_param, scaling)
#     evaluate(base_model, base_tokenizer,"neg")
#     c_attn_lora_method.remove_hooks(hooks)


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