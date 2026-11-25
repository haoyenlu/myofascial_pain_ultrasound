for f in 1 2 3 4 5; do python -m videomae.pretrain --from_scratch --fold "$f" --epochs 100 --batch_size 16 --lr 1e-3 --mask_ratio 0.9 --warmup_epochs 10 --grad_clip_norm 5.0 --num_workers 0; done
