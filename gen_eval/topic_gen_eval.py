import torch
from transformers import RobertaTokenizer
import json
import os
from .utils import generation, compute_ppl, compute_distinct,compute_topic_acc
@torch.no_grad()
def topic_gen_eval(args):
    output_path = args.output_path
    batch_size = args.batch_size
    os.makedirs(output_path, exist_ok=True)
    all_texts = generation(args)
    print("Evaluating...")
    ave_acc, label_list = compute_topic_acc(args, all_texts, batch_size, args.attr)
    ave_ppl, ppl_list = compute_ppl(args, all_texts, batch_size)
    dist1, dist2, dist3 = compute_distinct(all_texts)

    eval_labels = {
        0: "world",
        1: "sports",
        2: "business",
        3: "science"
    }
    file_name = f"{args.task}_{args.attr}_ACC_{ave_acc}_ppl_{ave_ppl}.jsonl"
    result_name = f"{args.task}_{args.attr}_result.json"
    with open(output_path + file_name, "w") as f:
        for text, ppl, attr in zip(all_texts, ppl_list, label_list):
            s = eval_labels[attr]
            obj = {"text": text, "ppl": ppl, "attr": s}
            json.dump(obj, f)
            f.write('\n')

    result = {
        "num_sequence": args.num_sequence,
        "length": args.generate_length,

        "task": args.task,
        "attr": args.attr,
        "acc":ave_acc,
        "ppl": ave_ppl,

        "dist-1": round(dist1, 4),
        "dist-2": round(dist2, 4),
        "dist-3": round(dist3, 4),
    }
    with open(output_path + result_name, "w") as f:
        json.dump(result, f, indent=2)
    print("Finished, Eval result has been saved in", output_path)





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/anke/DXZ/CTG_papers_code/Air-Decoding-main/models/best_topic_classifier"
    data_path = "/home/anke/DXZ/CTG_papers_code/Air-Decoding-main/test_data/128Air_prefix_topic.jsonl"

    batch_size = 16
    model = RobertaForPreTraining.from_pretrained(model_path).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

