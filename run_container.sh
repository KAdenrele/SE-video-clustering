docker run --rm -it \
  --gpus "device=0" \
  --shm-size=8g \
  --env-file .env \
  -v /mnt/data/test_dataset/raw/videos:/workspace/video_data \
  -v /home/ade/SE-video-clustering:/workspace/video_cluster \
  video_clustering