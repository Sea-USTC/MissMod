for i in `seq 0.2 0.2 1.6`;
do

    python  surf_md.py --data_root '/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race/' --dataset 'cefa'  --method 'md' --gpu 7 --version 1 --weight_sum 1.0 --weight_kld 0.8 --weight_con $i

done

