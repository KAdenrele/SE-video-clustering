docker build -t video_clustering . && docker run -it \
                                                  --name video-clustering-container \
                                                  --gpus "device=0" \
                                                  --shm-size=8g \
                                                  --env-file .env \
                                                  -v /mnt/data3/video_clustering/videos:/workspace/video_data \
                                                  -v /mnt/data/test_dataset/raw:/workspace/hf_cache \
                                                  -v /home/ade/SE-video-clustering:/workspace/video_cluster \
                                                  -v /mnt/data3/video_clustering/models:/workspace/models \
                                                  video_clustering