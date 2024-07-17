#!/usr/bin/env bash

python surf_baseline_multi_main.py --data_root '/remote-home/share/mengxichen/CASIA-SURF/' --drop_mode 'average' --backbone 'full_' --gpu 4 --version 0 --modal 'multi' --network 'student' --p [1,1,1] 
# python surf_baseline_multi_main.py --data_root '/GPFS/data/mengxichen/MMANet-CVPR2023-main/data/CASIA-SURF' --drop_mode 'average' --backbone 'full_' --gpu 5 --version 1 --modal 'multi' --network 'student' --p [0,0,0] 
# python surf_baseline_multi_main.py --data_root '/GPFS/data/mengxichen/MMANet-CVPR2023-main/data/CASIA-SURF' --drop_mode 'average' --backbone 'full_' --gpu 5 --version 2 --modal 'multi' --network 'student' --p [0,0,0] 