docker run --rm -it \
  --gpus all \
  -v /mnt/data/test_dataset/raw/video:/workspace/video_data \
  -v /home/ade/SE-video-clustering:/workspace/video_cluster \
  video_custering