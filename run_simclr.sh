for f in 1 2 3 4 5; do python -m simclr.pretrain --fold "$f"; done
python -m simclr.finetune --init_mode kinetics
