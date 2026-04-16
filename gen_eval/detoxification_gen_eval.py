
from .utils import compute_distinct, generation, compute_ppl, compute_toxicity
import json
import os
import torch


@torch.no_grad()
def detoxic_gen_eval(args):
    output_path = args.output_path
    batch_size = args.batch_size
    os.makedirs(output_path, exist_ok=True)

    all_texts = generation(args)
    print("Evaluating...")

    ave_toxicity, toxicity_list = compute_toxicity(args, all_texts, batch_size)
    ave_ppl, ppl_list = compute_ppl(args, all_texts, batch_size)
    dist1, dist2, dist3 = compute_distinct(all_texts)

    file_name = f"{args.task}_{args.attr}_TOX_{ave_toxicity}_ppl_{ave_ppl}.jsonl"
    result_name = f"{args.task}_{args.attr}_result.json"
    with open(output_path + file_name, "w") as f:
        for text, ppl, tox in zip(all_texts, ppl_list, toxicity_list):

            obj = {"text": text, "ppl": ppl, "toxicity": tox}
            json.dump(obj, f)
            f.write('\n')

    result = {
        "num_sequence": args.num_sequence,
        "length": args.generate_length,

        "task": args.task,
        "attr": args.attr,
        "toxicity": ave_toxicity,
        "ppl": ave_ppl,

        "dist-1": round(dist1, 4),
        "dist-2": round(dist2, 4),
        "dist-3": round(dist3, 4),
    }
    with open(output_path + result_name, "w") as f:
        json.dump(result, f, indent=2)
    print("Finished, Eval result has been saved in", output_path)