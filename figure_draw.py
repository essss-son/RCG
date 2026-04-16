import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification
import json
import torch
#note figure

def read_reward(path):
    rewards = []
    with open(path, 'r') as f:
        reward = json.load(f)
        for ite in reward:

            rewards.append(ite[-1])

    return rewards



def appendix_classifier_sent_degrad():

    save_path = "./figures/"
    # note sent
    x = [64,128,192,256,320,384,448,512]
    y1 = [92.5,96.66,98.33,97.08,94.58,97.08,95.0,92.08] # policy on
    y2 = [95.83,96.25,97.91,98.75,98.33,98.00,97.68,95.36]#wo RL
    # note topic
    # x = [64, 128, 192, 256, 320, 384, 448, 512]
    # y1 = [95.94,81.25,79.33,80.99,75.3,73.99,71.53,70.99]  # policy on
    # y2 = [95.62,95.31,96.22,97.19,95.62,97.5,96.25,95.02]  # wo RL
    # note detox
    # x = [64, 128, 192, 256, 320, 384, 448, 512]
    # y1 = [22.81,19.53,17.01,13.93,11.77,11.73,11.02,9.06]  # policy on
    # y2 = [18.97,20.16,20.44,20.26,19.91,20.49,20.42,20.71]  # wo RL


    fig, ax = plt.subplots()
    ax.plot(x, y1, label="no/chunk", marker="o")
    ax.plot(x, y2, label="chunk", marker="*")

    ax.set_xlabel("Eval length", fontsize=16)
    ax.set_ylabel("Eval Accuracy", fontsize=16)
    ax.legend(fontsize=15)
    ax.set_ylim(70, 100)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小
    ax.set_xticks(x)
    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "sent_class_degrad.png", dpi=300)



def appendix_figure9_activation_ablation_acc():

    save_path = "./figures/"

    x = [64,128,256,512]
    y1 = [95.83,98.54,98.33,95.83] # policy on
    y2 = [71.67,74.17,75.00,73.33]#wo RL
    y3 = [97.47,98.28,98.60,97.66]#wo fusion


    fig, ax = plt.subplots()
    ax.plot(x, y1, label="f/x", marker="o")
    ax.plot(x, y2, label="f/sigmoid", marker="*")
    ax.plot(x, y3, label="f/tanh", marker="^")



    ax.set_xlabel("Eval length", fontsize=16)
    ax.set_ylabel("Eval Accuracy", fontsize=16)
    ax.legend(fontsize=14)
    # plt.tight_layout()  # 调整整体布局

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小
    ax.set_xticks(x)
    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "figure9_activation_eval_acc.png", dpi=300)



def appendix_figure9_noise_ablation_acc():

    save_path = "./figures/"

    x = [64,128,256,512]
    y1 = [99.58,98.33,100.0,96.25] # policy on
    y2 = [92.91,95.00,97.10,92.10]#wo RL
    y3 = [97.33,98.59,97.83,96.08]#wo fusion
    y4 = [82.50,92.10,90.41,90.83]#wo gate
    y5= [95.83,98.75,98.75,94.58]#w/o Residual

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="no/noise", marker="o")
    ax.plot(x, y2, label="norm/noise", marker="*")
    ax.plot(x, y3, label="norm_decay/noise", marker="^")
    ax.plot(x, y4, label="rand/noise", marker="v")
    ax.plot(x, y5, label="rand_decay/noise", marker="v")


    ax.set_xlabel("Eval length", fontsize=16)
    ax.set_ylabel("Eval Accuracy", fontsize=16)
    ax.legend(fontsize=13.5)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小
    ax.set_xticks(x)
    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "figure9_noise_eval_acc.png", dpi=300)






def figure8_ablation_activation_dist():
    save_path = "./figures/"

    x = [5, 10, 15, 20, 25, 30, 35, 40,45,50]
    # note 找对应的acc
    y1 = [0.739,0.737,0.725,0.714,0.705,0.679,0.659,0.625,0.611,0.609] # policy on
    y2 = [0.739,0.744,0.718,0.71,0.698,0.665,0.627,0.613,0.589,0.513]#wo RL
    y3 = [0.734,0.739,0.748,0.738,0.707,0.683,0.657,0.629,0.624,0.608]#wo fusion

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="f/x", marker="v")
    ax.plot(x, y2, label="f/sigmoid", marker="*")
    ax.plot(x, y3, label="f/tanh", marker="o")




    ax.set_xlabel("Train Epoch", fontsize=16)
    ax.set_ylabel("Eval Dist-n", fontsize=16)
    ax.legend(fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "figure8_dist.png", dpi=300)



def figure8_ablation_activation_reward():
    x_noise = "/home/anke/DXZ/freectrl/reward_activation/x.json"
    sigmoid_decay = "/home/anke/DXZ/freectrl/reward_activation/tanh.json"
    tanh_decay = "/home/anke/DXZ/freectrl/reward_activation/sigmoid.json"


    r1 = read_reward(x_noise)
    r2 = read_reward(sigmoid_decay)
    r3 = read_reward(tanh_decay)



    save_path = "./figures/"

    x = range(1,len(r1)+1)

    fig, ax = plt.subplots()
    ax.plot(x, r1, label="f/x")
    ax.plot(x, r2, label="f/sigmoid")
    ax.plot(x, r3, label="f/tanh")


    ax.set_xlabel("Train Epoch", fontsize=16)
    ax.set_ylabel("Average Reward", fontsize=16)
    ax.legend(fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "figure8_reward.png", dpi=300)



def figure8_ablation_activation_speed():
    save_path = "./figures/"

    x = [5, 10, 15, 20, 25, 30, 35, 40,45,50]
    # note 找对应的acc
    y1 = [58.33,60.00,68.33,69.17,76.67,86.67,93.33,97.50,99.17,99.17] # policy on
    y2 = [53.33,65.83,66.67,80.00,70.00,73.33,76.67,77.50,81.67,89.17]#wo RL
    y3 = [58.3,62.5,69.17,80.83,92.5,90.83,96.67,97.50,95.00,98.33]#wo fusion

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="f/x", marker="o")
    ax.plot(x, y2, label="f/sigmoid", marker="*")
    ax.plot(x, y3, label="f/tanh", marker="^")


    ax.set_xlabel("Train Epoch", fontsize=16)
    ax.set_ylabel("Eval Accuracy", fontsize=16)
    ax.legend(fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "figure8_speed.png", dpi=300)










def figure7_c_ablation_noise_dist():
    save_path = "./figures/"

    x = [5, 10, 15, 20, 25, 30, 35, 40,45,50]
    # note 找对应的acc
    y1 = [0.732,0.732,0.695,0.66,0.616,0.507,0.458,0.416,0.374,0.300]
    y2 =[0.745,0.732,0.736,0.713,0.699,0.662,0.645,0.601,0.55,0.484]
    y3 =  [0.739,0.737,0.725,0.714,0.705,0.679,0.659,0.625,0.611,0.609]
    y4 =[0.734, 0.739, 0.734, 0.723, 0.685, 0.654, 0.615, 0.535, 0.406, 0.312]
    y5=  [0.739,0.738,0.735,0.727,0.708,0.706,0.686,0.636,0.58,0.556]
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="no/noise", marker="o")
    ax.plot(x, y2, label="norm/noise", marker="*")
    ax.plot(x, y3, label="norm_decay/noise", marker="^")
    ax.plot(x, y4, label="rand/noise", marker="v")
    ax.plot(x, y5, label="rand_decay/noise", marker="v")



    ax.set_xlabel("Train Epoch", fontsize=16)
    ax.set_ylabel("Eval Dist-n", fontsize=16)
    ax.legend(fontsize=15)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "noise_ablation_dist.png", dpi=300)




def figure7_b_ablation_noise_speed():
    save_path = "./figures/"

    x = [5, 10, 15, 20, 25, 30, 35, 40,45,50]
    # note 找对应的acc
    y1 = [67.50,76.67,90.00,98.33,99.17,98.33,97.50,98.33,96.67,92.50] # policy on
    y2 = [59.17,69.17,85.00,95.00,96.67,98.33,100,99.17,100,100]#wo RL
    y3 = [61.7,69.17,74.17,88.33,92.50,95.83,98.33,99.17,100,99]#wo fusion
    y4 = [58.33,60.00,68.33,69.17,76.67,86.67,93.33,97.50,99.17,99.17]#wo gate
    y5= [57.50,73.33,78.33,88.33,92.50,98.33,96.67,100,99.17,100]#w/o Residual
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="no/noise", marker="o")
    ax.plot(x, y2, label="norm/noise", marker="*")
    ax.plot(x, y3, label="norm_decay/noise", marker="^")
    ax.plot(x, y4, label="rand/noise", marker="v")
    ax.plot(x, y5, label="rand_decay/noise", marker="v")



    ax.set_xlabel("Train Epoch", fontsize=16)
    ax.set_ylabel("Eval Accuracy", fontsize=16)
    ax.legend(fontsize=13.5)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "noise_ablation_speed.png", dpi=300)





def figure7_a_ablation_noise_reward():
    no_noise = "/home/anke/DXZ/freectrl/noise_ablation_reward/no_noise.json"
    norm_decay = "/home/anke/DXZ/freectrl/noise_ablation_reward/norm_decay.json"
    norm_no_decay = "/home/anke/DXZ/freectrl/noise_ablation_reward/norm_no_decay.json"
    rand_decay = "/home/anke/DXZ/freectrl/noise_ablation_reward/rand_decay.json"
    rand_no_decay = "/home/anke/DXZ/freectrl/noise_ablation_reward/rand_no_decay.json"

    r1 = read_reward(no_noise)
    r2 = read_reward(norm_decay)
    r3 = read_reward(norm_no_decay)
    r4 = read_reward(rand_decay)
    r5 = read_reward(rand_no_decay)


    save_path = "./figures/"

    x = range(1,len(r1)+1)

    fig, ax = plt.subplots()
    ax.plot(x, r1, label="no/noise")
    ax.plot(x, r2, label="norm/noise")
    ax.plot(x, r3, label="norm_decay/noise")
    ax.plot(x, r4, label="rand/noise")
    ax.plot(x, r5, label="rand_decay/noise")


    ax.set_xlabel("Train Epoch", fontsize=16)
    ax.set_ylabel("Average Reward", fontsize=16)
    ax.legend(fontsize=13.5)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "noise_ablation_reward.png", dpi=300)





def ablation_figure46():

    save_path = "./figures/"



    #note eval length
    x = [64,128,256,512]
    y1 = [97.435,99.08,99.23,97.36] # policy on
    y2 = [68.75,66.66,70.415,64.79]#wo RL
    y3 = [64.375,70.41,68.33,64.165]#wo fusion
    y4 = [96.354,99.375,97.91,78.33]#wo gate
    y5= [96.435,97.08,96.34,96.5]#w/o Residual

    #note eval accuracy
    # x = [5,10,15,20,25,30,35,40,45,50]
    # y1 = [68.5, 78.0, 87.0, 93.2, 98.5, 98.7, 97.9, 98.4, 96.4, 98.5]  # policy on
    # y2 = [68.1,64.6,71.3,67.4,74.5,75.9,71.8,73.4,69.5,66.4]  # wo RL
    # y3 = [71.5,61.8,62.9,68.5,66.2,60.9,70.44,64.3,62.6,72.5]  # wo fusion
    # y4 = [72.5, 90.0, 98.3, 98.1, 98.4, 99.10, 97.28, 99.30, 98.88, 99.20]  # wo gate
    # y5 = [62.5,64.5,63.8,69.3,75.4,79.6,86.7,92.6,96.5,97.8]  # w/o Residual



    fig, ax = plt.subplots()
    ax.plot(x, y1, label="policy-on", marker="o")
    ax.plot(x, y2, label="w/o RL", marker="*")
    ax.plot(x, y3, label="w/o Fusion", marker="^")
    ax.plot(x, y4, label="w/o Gate", marker="v")
    ax.plot(x, y5, label="w/o Residual", marker=">")


    ax.set_xlabel("Eval Length", fontsize=15)
    ax.set_ylabel("Eval Accuracy", fontsize=15)
    ax.legend()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小
    ax.set_xticks(x)
    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "module_ablation_acc.png", dpi=300)




def fig1():
    save_path = "./figures/"

    x = np.array([64, 128, 192, 256, 320, 384, 448, 512])
    y1 = np.array([99.16,99.16,99.50,98.75,98.95,97.20,98.00,97.50])  # RCG (Ours)
    y2 = np.array([93.75, 92.91, 90.33, 82.91, 80.89, 80.12, 73.3, 70])      # Air-decoding
    y3 = np.array([92.8, 88.62, 87.63, 83.66, 81.55, 80.36, 80.1, 77.98])    # SF-GEN

    fig, ax = plt.subplots()

    ax.plot(x, y1, marker="o", label="RCG (Ours)", color="red")
    ax.plot(x, y2, marker="*", label="Air-decoding", color="purple")
    ax.plot(x, y3, marker="^", label="SF-GEN", color="green")

    ax.set_xlabel("Length", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.legend(fontsize=13)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xticks(x)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.minorticks_on()

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', which='minor', length=3)

    # 添加箭头
    ax.annotate(
        '', xy=(1.02, 0), xycoords='axes fraction', xytext=(1.0, 0),
        arrowprops=dict(arrowstyle='->', linewidth=1.5)
    )
    ax.annotate(
        '', xy=(0, 1.02), xycoords='axes fraction', xytext=(0, 1.0),
        arrowprops=dict(arrowstyle='->', linewidth=1.5)
    )

    plt.tight_layout()
    plt.savefig(save_path + "fig1.png", dpi=300)
    plt.close()





def heatmap_figure3():
    import matplotlib.pyplot as plt
    import numpy as np


    score = [0.388, 0.596, 0.936, 0.983, 0.993, 0.957, 0.999, 0.996, 0.991, 0.999,
             0.995, 0.999, 0.998, 0.999, 0.999, 0.999, 0.997, 0.999, 0.997, 0.999]

    hid_norm = [1.711, 1.742, 1.763, 1.781, 1.792,
                1.704, 1.722, 1.711, 1.735, 1.742,
                1.756, 1.747, 1.764, 1.772, 1.732,
                1.796, 1.705, 1.724, 1.736, 1.752]

    strength = [0.860, 0.690, 0.610, 0.460, 0.432,
                0.540, 0.482, 0.482, 0.490, 0.456,
                0.470, 0.463, 0.450, 0.401, 0.452,
                0.390, 0.470, 0.458, 0.439, 0.414]

    steps = np.arange(1, 21)

    # 2. 设置画布风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 8))

    # 3. 绘制散点图 (热力点)
    sc = ax.scatter(steps, hid_norm, c=strength, cmap='RdYlBu_r', s=200,
                    edgecolors='black', alpha=0.9, zorder=10)

    # 4. 绘制趋势线
    ax.plot(steps, hid_norm, color='gray', linestyle='--', alpha=0.4, zorder=1)

    # 5. 添加 Score 标注
    for i in range(len(steps)):
        label = f"{score[i]}"
        ax.annotate(label,
                    xy=(steps[i], hid_norm[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=18,
                    # fontweight='bold',
                    color='#333333',
                    rotation=0)

    ax.set_xlabel('Steps', fontsize=30)
    ax.set_ylabel('Hidden Norm', fontsize=30 )

    # 2. 设置刻度数字 (Ticks) - 字号 15
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 3. 设置大标题 - 字号 20
    # ax.set_title('Hidden Norm Trend with Strength Heatmap', fontsize=20, pad=20)


    # 6. 设置坐标轴 (使用英文)
    ax.set_xlabel('Steps', fontsize=23, )
    # ax.set_ylabel('Attribute Representation Norm', fontsize=24)
    ax.set_ylabel(r'$\|\mathbf{V}_{\mathrm{ctrl}}\|$', fontsize=24)    # ax.set_title('Hidden Norm Trend with Strength Heatmap', fontsize=16, pad=20)

    # 设置X轴刻度
    ax.set_xticks(steps)
    ax.set_xlim(0.5, 20.5)

    # 自动调整Y轴范围
    y_min, y_max = min(hid_norm), max(hid_norm)
    margin = (y_max - y_min) * 0.2
    ax.set_ylim(y_min - margin * 0.5, y_max + margin)

    # 7. 添加右侧热力对照条 (Colorbar)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('$\\beta_t$', fontsize=30, labelpad=15)
    cbar.outline.set_visible(False)

    # 8. 保存或显示
    plt.tight_layout()
    # 建议直接保存，因为在服务器终端可能无法直接弹窗显示
    plt.savefig('./figures/heat_plot.png', dpi=300)
    print("Plot saved to heat_plot.png")


def lora_loss():
    # neg_loss = "/home/anke/DXZ/freectrl/Loss_neg.json"
    # r1 = read_reward(neg_loss)

    # world_loss = "/home/anke/DXZ/freectrl/Loss_world.json"
    # r1 = read_reward(world_loss)

    tox_loss = "/home/anke/DXZ/freectrl/Loss_nontoxic.json"
    r1 = read_reward(tox_loss)

    save_path = "./figures/"

    x = range(1, len(r1) + 1)

    fig, ax = plt.subplots()
    ax.plot(x, r1)
    # ax.plot(x, r2, label="norm/noise")
    # ax.plot(x, r3, label="norm_decay/noise")
    # ax.plot(x, r4, label="rand/noise")
    # ax.plot(x, r5, label="rand_decay/noise")

    ax.set_xlabel("Train Step", fontsize=16)
    ax.set_ylabel("Loss Value", fontsize=16)
    ax.legend(fontsize=13.5)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "toxic_loss.png", dpi=300)


def gate_reward_figure5():

    tox_loss = "/home/anke/DXZ/RCG/figures/only_gate_reward_data.json"
    r1 = read_reward(tox_loss)
    r1 = [f-1 for f in r1]
    save_path = "./figures/"

    x = range(1, len(r1) + 1)

    fig, ax = plt.subplots()
    ax.plot(x, r1)

    ax.set_xlabel("Train Epoch", fontsize=15)
    ax.set_ylabel("Average Reward", fontsize=15)
    ax.legend(fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis='x', labelsize=14)  # 横轴数字大小
    ax.tick_params(axis='y', labelsize=14)  # 纵轴数字大小

    ax.annotate(
        "",
        xy=(1.02, 0), xycoords="axes fraction",
        xytext=(1.0, 0), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path + "only_gate_reward.png", dpi=300)




if __name__ == "__main__":
    ablation_figure46()