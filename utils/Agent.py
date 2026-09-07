import json
from torch.optim import AdamW
from .utils import reward_fn, c_attn_lora_method, get_injected_model, MyDataset,ppo_collate
import torch
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from itertools import chain
from transformers import GPT2ForSequenceClassification,GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer,RobertaTokenizer,RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
#note IDE
# from version_9.model_utils.model_topic import RobertaForPreTraining
# from version_9.model_utils.model import Critic

#note 跑脚本
# from model_utils.model import Critic
from model_utils.model import Critic

class Agent:
    def __init__(self,args):
        self.args = args
        self.device = args.device

        self.gpt2 = self.get_train_model(args.attr)

        self.policy = self.gpt2.transformer
        self.lm_head = self.gpt2.lm_head

        self.ref_gpt2, self.tokenizer = c_attn_lora_method.get_gpt2()
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_model, self.reward_tokenizer = self.get_reward_model_tokenizer(args.task)
        # for name, p in self.gpt2.named_parameters():
        #     print(name, p.requires_grad)
        self.actor_optimizer = AdamW(filter(lambda p:p.requires_grad,self.gpt2.parameters()), lr=args.actor_lr,weight_decay=5e-4)

        self.critic_base,_ = c_attn_lora_method.get_gpt2()
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
            # model = DistilBertForSequenceClassification.from_pretrained(
            #     "/home/anke/DXZ/models/distilbert-base-uncased-finetuned-sst-2-english").to(self.device).eval()
            # tokenizer = DistilBertTokenizer.from_pretrained("/home/anke/DXZ/models/distilbert-base-uncased-finetuned-sst-2-english")

            tokenizer.pad_token = tokenizer.eos_token
        elif task == "topic":
            model = RobertaForSequenceClassification.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news").to(self.device).eval()
            tokenizer = RobertaTokenizer.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news")
        elif task == "detoxification":
            model = AutoModelForSequenceClassification.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta").to(self.device).eval()
            tokenizer = AutoTokenizer.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta")
        elif task == "multi":

            attr_list = self.args.attr
            model_set = set()
            for attr in attr_list:
                if attr in ["pos","neg"]:
                    model_set.add("sentiment")
                elif attr in ["world","sports","business","science"]:
                    model_set.add("topic")
                elif attr in ["nontoxic"]:
                    model_set.add("detoxification")
            if "sentiment" in model_set:
                sent_model = GPT2ForSequenceClassification.from_pretrained("/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment").to(self.device).eval()
                sent_tokenizer = GPT2Tokenizer.from_pretrained("/home/anke/DXZ/models/gpt2-medium-finetuned-sst2-sentiment")
            if "topic"  in model_set:
                topic_model = RobertaForSequenceClassification.from_pretrained(
                    "/home/anke/DXZ/models/roberta-based-ag-news").to(self.device).eval()
                topic_tokenizer = RobertaTokenizer.from_pretrained("/home/anke/DXZ/models/roberta-based-ag-news")
            if "detoxification" in model_set:
                detoxic_model = AutoModelForSequenceClassification.from_pretrained(
                    "/home/anke/DXZ/models/unbiased-toxic-roberta").to(self.device).eval()
                detoxic_tokenizer = AutoTokenizer.from_pretrained("/home/anke/DXZ/models/unbiased-toxic-roberta")

        if model is None or tokenizer is None:
            raise Exception("No model or tokenizer found")
        return model, tokenizer

    def get_train_model(self,attr):
        model = None
        # if len(attr)==1:
        model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=len(attr))
        for controller in model.fusion_controllers:
            controller.set_active_adapters(attr)
            # if attr == "pos":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['pos'])
            #
            # elif attr == "neg":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['neg'])
            # elif attr == "world":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['world'])
            # elif attr == "sports":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['sports'])
            # elif attr == "business":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['business'])
            # elif attr == "science":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['science'])
            #
            # elif attr == "nontoxic":
            #     model = get_injected_model(self.args.lora_path_dict, self.device, num_active_adapters=1)
            #     for controller in model.fusion_controllers:
            #         controller.set_active_adapters(['nontoxic'])
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
            "sentiment":"/home/anke/DXZ/RCG/dataset/sentiment-imdb/prompt_sent.jsonl",
            "topic":"/home/anke/DXZ/RCG/dataset/topic-agnews/prompt_topic.jsonl",
            "detoxification":"/home/anke/DXZ/RCG/dataset/detoxification-jigsaw/prompt_detoxification.jsonl",
            "multi": "/home/anke/DXZ/RCG/dataset/sentiment-imdb/prompt_sent.jsonl",
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
            decay = 0.999
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