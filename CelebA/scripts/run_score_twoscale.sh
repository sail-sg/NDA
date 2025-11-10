echo "gpu_ids:          $1"
echo "setting:          $2"
echo "start:            $3"
echo "end:              $4"
echo "t_fixed:          $5"
echo "patch_size:       $6"
echo "topk:             $7"
echo "patch_size_down:  $8"
echo "alpha:            $9"
echo "gen_source:       ${10}"
echo "mask_value:       ${11}"
echo "kernel_batch_size:${12}"

PROJ_DIM=$(( $8 / 2 ))

OUT_BASE="./results/$2/twoscale/${10}-mask${11}/topk$7/t=$5/ps=$6_psd=$8_a=$9"
VIS_DIR="${OUT_BASE}/vis"
mkdir -p "${VIS_DIR}"

CUDA_VISIBLE_DEVICES="$1" python nda_twoscale.py \
  --dataset_name "celeba" \
  --dataset_config_name "" \
  --index_path "./data/indices/$2/idx-train.pkl" \
  --gen_path   "./saved/$2/gen" \
  --gen_start "$3" \
  --gen_end "$4" \
  --t_fixed "$5" \
  --patch_size "$6" \
  --weight_topk "$7" \
  --patch_size_down "$8" \
  --alpha "$9" \
  --gen_source "${10}" \
  --mask_value "${11}" \
  --kernel_batch_size "${12}" \
  --output_dir "${OUT_BASE}" \
  --save_vis_dir "${VIS_DIR}" \
  --t_strategy "cumulative" \
  --proj_dim "${PROJ_DIM}" \
  --K 10 \
  --e_seed 0 \
  --resolution 64 \
  --train_batch_size 512 \
  --ddpm_num_steps 1000 \
  --ddpm_beta_schedule "linear" \
  --dataloader_num_workers 8
