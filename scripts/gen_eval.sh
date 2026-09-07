#pos_arg=/home/anke/DXZ/RCG/rl_train/sentiment/pos/v2/hyper_params.json
#pos_w=/home/anke/DXZ/RCG/rl_train/sentiment/pos/v2/checkpoints/Epoch40_ACC_98.3_ppl17.9
#
#neg_arg=/home/anke/DXZ/RCG/rl_train/sentiment/neg/v2/hyper_params.json
#neg_w=/home/anke/DXZ/RCG/rl_train/sentiment/neg/v2/checkpoints/Epoch50_ACC_95.8_ppl18.1

world_arg=/home/anke/DXZ/RCG/rl_train/topic/world/v2/hyper_params.json
world_w=/home/anke/DXZ/RCG/rl_train/topic/world/v2/checkpoints/Epoch40_ACC_99.4_ppl18.8

#sports_arg=/home/anke/DXZ/RCG/rl_train/topic/sports/v2/hyper_params.json
#sports_w=/home/anke/DXZ/RCG/rl_train/topic/sports/v2/checkpoints/Epoch50_ACC_100.0_ppl18.3
#
#business_arg=/home/anke/DXZ/RCG/rl_train/topic/business/v2/hyper_params.json
#business_w=/home/anke/DXZ/RCG/rl_train/topic/business/v2/checkpoints/Epoch40_ACC_99.4_ppl18.1
#
#science_arg=/home/anke/DXZ/RCG/rl_train/topic/science/v2/hyper_params.json
#science_w=/home/anke/DXZ/RCG/rl_train/topic/science/v2/checkpoints/Epoch50_ACC_98.8_ppl19.9

#nontoxic_arg=/home/anke/DXZ/RCG/rl_train/detoxification/nontoxic/v2/hyper_params.json
#nontoxic_w=/home/anke/DXZ/RCG/rl_train/detoxification/nontoxic/v2/checkpoints/Epoch50_Toxicity_33.9_ppl14.3

lengths=(64 128 256 512)
#lengths=(64 64 64 64)
batch_sizes=(16 16 4 2)


#for i in "${!lengths[@]}"
#do
#  python generation_eval.py \
#    --args_path "$pos_arg" \
#    --policy_path "$pos_w" \
#    --batch_size "${batch_sizes[$i]}" \
#    --num_sequence 16 \
#    --generate_length "${lengths[$i]}"
#done
#
#for i in "${!lengths[@]}"
#do
#  python generation_eval.py \
#    --args_path "$neg_arg" \
#    --policy_path "$neg_w" \
#    --batch_size "${batch_sizes[$i]}" \
#    --num_sequence 16 \
#    --generate_length "${lengths[$i]}"
#done

for i in "${!lengths[@]}"
do
  python generation_eval.py \
    --args_path "$world_arg" \
    --policy_path "$world_w" \
    --batch_size "${batch_sizes[$i]}" \
    --num_sequence 16 \
    --generate_length "${lengths[$i]}"
done

#for i in "${!lengths[@]}"
#do
#  python generation_eval.py \
#    --args_path "$sports_arg" \
#    --policy_path "$sports_w" \
#    --batch_size "${batch_sizes[$i]}" \
#    --num_sequence 16 \
#    --generate_length "${lengths[$i]}"
#done
#
#for i in "${!lengths[@]}"
#do
#  python generation_eval.py \
#    --args_path "$business_arg" \
#    --policy_path "$business_w" \
#    --batch_size "${batch_sizes[$i]}" \
#    --num_sequence 16 \
#    --generate_length "${lengths[$i]}"
#done
#
#for i in "${!lengths[@]}"
#do
#  python generation_eval.py \
#    --args_path "$science_arg" \
#    --policy_path "$science_w" \
#    --batch_size "${batch_sizes[$i]}" \
#    --num_sequence 16 \
#    --generate_length "${lengths[$i]}"
#done


#nlengths=(64 128 256 512)
#nbatch_sizes=(2 2 2 2)
#
#
#for i in "${!nlengths[@]}"
#do
#  python generation_eval.py \
#    --args_path "$nontoxic_arg" \
#    --policy_path "$nontoxic_w" \
#    --batch_size "${nbatch_sizes[$i]}" \
#    --num_sequence 2 \
#    --generate_length "${nlengths[$i]}"
#done
