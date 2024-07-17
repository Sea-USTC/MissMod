for i in `seq 0.2 0.2 2`;
do

    python  surf_etmc.py --data_root '/remote-home/share/mengxichen/CASIA-SURF/' --dataset 'surf'  --method 'etmc' --gpu 1 --version 1 --weight_etmc $i

done

