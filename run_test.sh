
INPUT_DIR='./test_img/in'
OUTPUT_DIR='./test_img/out'
CHECK_GAN='./checkpoints/gan'
CHECK_RESNET='./checkpoints/resnet/weights_resnet_101_128x128_batch_128.hdf5'
CHECK_DEEPLAB='/checkpoints/deeplab'
EMBED='./checkpoints/w2v/coco_embeddings.json'
CLASS_MAP='./checkpoints/w2v/class_map.json'
CLASS_NAMES='./checkpoints/w2v/things_stuff_labels.txt'
PASCAL_DATASET="./dataset/tfrecord"
VIS_LOGDIR='./test_img/out/vis'

# Visualize the results.
python "${WORK_DIR}"./occlusion.py \
  --logtostderr \
  --dataset="ms_coco_stuff" \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=660 \
  --vis_crop_size=660 \
  --checkpoint_deeplab="${CHECK_DEEPLAB}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_iterations=1 \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_gan="${CHECK_GAN}" \
  --checkpoint_resnet="${CHECK_RESNET}" \
  --input_dir="${INPUT_DIR}" \
  --embeddings="${EMBED}" \
  --class_map_w2v="${CLASS_MAP}" \
  --class_names="${CLASS_NAMES}"
