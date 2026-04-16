version_9
# note novelty 强化学习代价小；一种policy所有长度通用(前提是训练的policy不能训练太久)；参数小；
1c_attn->mlp看看效果

2要提到的：1参数轻量化，跟baseline对比参数；强化学习代价小，短epoch，多长度通用；

3尝试说明在强化学习中lora相对于prefix的优越性
prefix方法的前部分生成很容易重复，究其原因是在相同的prompt下注意力机制的固化问题，相同prefix+相同prompt导致了相同的注意力模式，使刚开始的生成缺乏多样性

4提到门控机制的灵活性，当属性已对齐时，门控会输出较小的控制强度（考虑是否实验？）

5需要说明在air-decoding里单纯将prefix换成lora模块后的对比测试效果，

6在局限里分析当policy训练过长之后出现的问题，可以摘取部分文本

7训练出来的ppl非常低，在论文里面可以着重说明，长文本的ppl同样很低

8我们选择gpt2的原因是为了证明在能力不强的模型下，我们的方法可以实现有效控制，因为更大的模型基座能力就更强（但是好像又不太合理，我们的要求又不会体现在prompt中）

9要分析为什么air-decoding在长度增加的时候acc就下降了，为什么我们的方法就不会？

10仅仅用残差流来进行长文本生成，结合强化学习模块进行长文本生成，第一种情况一定会属性下降，第二种情况不会，说明rl的必要性（思考怎么说），毕竟训练代价很小

11展望：还有什么可以提高的点，我们会在其他更大的模型上采用该机制

12长文本生成的应用；贡献里面：短文本也有很高的精度，但主要贡献在长文本



# note test to append
1去掉重复惩罚，把kl换成噪声，收集长度换成512的pos测下数据


#note problem solving
解决的问题：
gate的激活函数会导致其输出取不到0，当policy崩坏后无法关门，改变激活函数,刚开始是tanh，现在改成了k sigmoid

#note writing
行文思路：
Abstract
受控属性生成->短文本效果好->长文本属性下降->如果要维持会增大控制强度导致属性崩溃->我们的方法不会导致属性崩溃，还能稳定acc，简要介绍
补充选全量微调和部分微调和decoding-time的方法和我们的对比，说明长文本下我们方法的优越性

Introduction:
介绍CTG
介绍3种方法
长文本生成本身的困难（属性漂移）->迫使现有方法（解码端）加大力度->导致分布破坏（属性崩溃）。同时，并行指出训练端方法（全参数/轻量级）在长文本上的局限性



#note model version
v8
门控可以分别控制pos-neg的残差流和fuse流

#note test to do
待测试任务：
7右padding加取最后一个token预测不管用？？？？测试
9检查全文计算ppl的时候考没考虑padding
10尝试解决用prompt生成时的长度不一致问题，解决后将prompt回归来的长度


#note Task to do
必做：
1如果要说噪声，补充噪声相关的消融实验（1）噪声随步数衰减（2）噪声的均值为0的时候  加1的时候
2补充关于属性奖励的激活函数的消融实验
3补充



3methodology描述框架
3.1关于lora，初训练
介绍lora，在主要方法里，我们选择gpt2的qkv_projection_matrix（c_attn）作为lora微调对象，对于每个属性我们各自训练了一个lora模块，得到了一共8个lora模块
3.2policy
我们的policy由一个信号融合网络和一个门控网络组成，信号融合网络决定对隐状态偏移的方向，门控网络根据当前文本状态，决定控制的强度，若当前已生成文本已经满足了属性要求，我们希望控制强度降低，让模型更流畅的输出。如果还不满足属性要求，我们希望动态增大控制强度，引导模型来生成更具有对应属性的文本（其他方法若想在长文本下保持属性往往要人为施加更大的控制强度来实现属性对齐，这往往会造成生成的文本可读性差，流畅性下降）。

3.2强化学习
3.2.1强化学习建模
概述状态、环境、动作的设计（参考相关论文）
3.2.2奖励设计（重点介绍一下）
reward=属性得分 + 参考模型logits引导 + 重复惩罚 - kl散度
我们通过恰当的奖励设置，实现了在保持流畅性下的属性对齐。
介绍每项奖励的构成。当生成了t个token时，将完整t个token输入对应奖励模型，得到属性得分奖励，越符合对应属性奖励越大。将t-1个token输入参考模型，取参考模型对第t个token的预测logits，越大说明越符合参考模型的分布，更大程度上满足了句法结构避免了语法错误。kl散度作为辅助，约束模型输出不要距离参考模型太远。重复惩罚引导使产生的文本更加多样性。
属性得分让policy学会让文本和属性对齐，基础模型logits引导和kl散度实现了让policy不要一味去追求属性得分，强行生成不符合句法规则的token会收到较重惩罚。在这样的奖励设计下，如果当前文本属性已经对齐（属性得分较高），policy会朝着增大基础模型Logits的方向优化，驱使门控网络减小控制强度，从而让policy网络学会有的放矢。


3.2.3强化学习训练（简单介绍？）
优化算法，ppo算法

3.3工作流
以sentiment任务的positive控制为例，在每一层gpt2 block里,文本编码过后得到input[batch_size, seq_len, hid_dim]分别输入给c_attn,pos_lora,neg_lora,分别得到形状相同[batch_size, seq_len, 3 * hid_dim]的输出base_output,pos_lora_output=pos_lora(input),neg_lora_output=neg_lora(input)，将其拼接起来得到policy的输入policy_input,其形状为[batch_size, seq_len, 9 * hid_dim];在这里policy由一个信号融合网络以及一个门控网络组成（在后文介绍其消融实验），信号融合网络接受policy_input，输出outputA=Fuse_net(policy_input) + residual,形状为[batch_size, seq_len, 3 * hid_dim],这里的residual来自于lora模块的输出在对应属性方向上做差(pos_lora_output - neg_lora_output),门控网络接受policy_input,输出outputB=Gate_net(policy_input),其形状为[batch_size, seq_len, 1]。作为控制强度信号，policy输出为outputC = outputA * outputB,形状为[batch_size, seq_len, 3 * hid_dim],最终输出Final_output=base_output + outputC.


尝试优化一下baselogits的数值，考虑使用sigmoid?


#note phenomeno
发现的一些有趣的现象：
2如果是单残差流的topic模型，属性打分权重是不是应该提高一些？



