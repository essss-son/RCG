import torch.nn as nn
import torch
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

        # MODIFIED: 动态计算输入维度
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
        # MODIFIED: 校验输入数量
        if len(adapter_outputs) != (self.input_multiplier - 1):
            raise ValueError(f"Policy expected {self.input_multiplier - 1} adapters, got {len(adapter_outputs)}")

        # 1. 拼接: [Batch, Seq, Dim * (1+N)]
        cat_input = torch.cat([base_out] + adapter_outputs, dim=-1)

        # 2. 计算残差基准
        # 原逻辑是 (pos - neg)。但在通用逻辑下，我们需要一个更通用的方式。
        # 方案：简单的求和作为基准，或者根据任务类型特化。
        # 这里使用 Sum(Adapters) 作为残差基准
        if len(adapter_outputs) == 2:
            residual_base = adapter_outputs[0] - adapter_outputs[1]

            # direction = self.net(cat_input) + (adapter_outputs[0] - adapter_outputs[1]) # 如果非要保留Pos-Neg
        elif len(adapter_outputs) == 1:
            residual_base = adapter_outputs[0]
        else:
            raise ValueError("残差输入出错，或者检查num_active_adapters")


        # # note final
        direction = self.net(cat_input) + residual_base
        alpha = self.final_gate(self.gate_net(cat_input) * 6)
        return direction * alpha

        # note final_fusion
        # direction = self.net(cat_input) + residual_base
        # return direction

        # # note final_gate
        # direction = residual_base
        # alpha = self.final_gate(self.gate_net(cat_input) * 6)
        # return direction * alpha

        # note only_lora
        # direction = residual_base
        # return direction




class LayerController(nn.Module):
    def __init__(self, layer_idx, hidden_size, qkv_output_dim,
                 all_adapters_params, all_scalings, num_active_adapters=2):
        """
        all_adapters_params: dict, { "pos": {'A':..., 'B':...}, "world": ... }
        all_scalings: dict, { "pos": 1.0, "world": 2.0 ... }
        num_active_adapters: 初始化 Policy 时设定的同时激活数量
        """
        super().__init__()
        self.layer_idx = layer_idx

        # MODIFIED: 使用 ParameterDict 存储所有 LoRA 参数 (LoRA Bank)
        self.lora_A_bank = nn.ParameterDict()
        self.lora_B_bank = nn.ParameterDict()
        self.scalings = {}  # scaling 是浮点数，直接存 dict 即可

        # 遍历注入所有 adapter
        for name, params in all_adapters_params.items():
            # 冻结参数
            self.lora_A_bank[name] = nn.Parameter(params['A'], requires_grad=False)
            self.lora_B_bank[name] = nn.Parameter(params['B'], requires_grad=False)
            self.scalings[name] = all_scalings[name]

        # MODIFIED: 状态变量，控制当前 forward 使用哪些 adapter
        self.active_adapter_names = []

        # 实例化 Policy
        self.policy = FusionPolicy(hidden_size, qkv_output_dim, num_active_adapters)

    def set_active_adapters(self, adapter_names):
        """
        在推理/训练前调用，指定当前 batch 要激活的 adapter 列表。
        列表顺序很重要，对应 Policy 的输入顺序。
        例如: ['pos', 'neg'] 或 ['world']
        """
        for name in adapter_names:
            if name not in self.lora_A_bank:
                raise ValueError(f"Adapter '{name}' not found in layer {self.layer_idx}")
        self.active_adapter_names = adapter_names

    def compute_lora(self, x, name):
        # 从 Bank 取参数
        A = self.lora_A_bank[name]
        B = self.lora_B_bank[name]
        scaling = self.scalings[name]

        x = x.to(A.dtype)
        return (x @ A.t()) @ B.t() * scaling

    def hook_fn(self, module, input, output):
        x = input[0]
        base_out = output

        # MODIFIED: 根据 active_names 动态计算
        adapter_outputs = []
        for name in self.active_adapter_names:
            out = self.compute_lora(x, name)
            adapter_outputs.append(out)

        # 送入 Policy
        # 注意：adapter_outputs 的数量必须匹配 Policy 初始化时的 num_active_adapters
        # fusion_delta = self.policy(base_out, adapter_outputs)
        fusion_delta = self.policy(base_out, [sum(adapter_outputs)])

        return base_out + fusion_delta




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



if __name__ == '__main__':
    pass