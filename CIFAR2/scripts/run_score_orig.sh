echo "gpu_ids:          $1"
echo "setting:          $2"
echo "start:            $3"
echo "end:              $4"
echo "t_fixed:          $5"
echo "patch_size:       $6"
echo "topk:             $7"
echo "gen_source:       $8"
echo "mask_value:       $9"
echo "kernel_batch_size:${10}"

OUT_BASE="./results/$2/sum/$8-mask$9/topk$7/t=$5/ps=$6"
VIS_DIR="${OUT_BASE}/vis"
mkdir -p "${VIS_DIR}"

CUDA_VISIBLE_DEVICES="$1" python nda_single_scale.py \
  --dataset_name "cifar10" \
  --dataset_config_name "" \
  --index_path "./data/indices/$2/idx-train.pkl" \
  --gen_path "./saved/$2/gen" \
  --gen_start "$3" \
  --gen_end "$4" \
  --t_fixed "$5" \
  --patch_size "$6" \
  --weight_topk "$7" \
  --gen_source "$8" \
  --output_dir "${OUT_BASE}" \
  --save_vis_dir "${VIS_DIR}" \
  --t_strategy "cumulative" \
  --mask_value "$9" \
  --K 10 \
  --e_seed 0 \
  --resolution 32 \
  --train_batch_size 512 \
  --ddpm_num_steps 1000 \
  --ddpm_beta_schedule "linear" \
  --dataloader_num_workers 8 \
  --kernel_batch_size "${10}"
