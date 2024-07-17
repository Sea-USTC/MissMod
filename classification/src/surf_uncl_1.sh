# for i in `seq 0.1 0.1 4`;
# do

#     # python  surf_uncl_teacher.py --data_root '/remote-home/share/mengxichen/CASIA-SURF/' --dataset 'surf'  --method 'uncl' --gpu 6 --version 1 --lambda_kd_shared $i --lambda_kd_feature 2.0
#     python  surf_uncl_teacher.py --data_root '/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race/' --dataset 'cefa' --method 'test' --gpu 7 --version 1 --lambda_kd_shared 1.8 --lambda_kd_feature 1.8 --train_epoch 1000 --total_epoch 50 --save_interval 60 --batch_size 512 --seed 3407

# done

# for i in `seq 1.4 0.2 1.6`;
# do
    # python  surf_uncl_teacher.py --data_root '/DB/public/mengxichen/CASIA-SURF' --dataset 'surf'  --method 'cat-2fc(0.5)-2' --gpu 7 --version 10 --lambda_kd_shared 2.0 --lambda_kd_feature 2.0 --train_epoch 110 --total_epoch 50 --save_interval 60 --seed 3407
    python  surf_uncl_teacher.py --data_root '/DB/public/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race' --dataset 'cefa'  --method 'cat-2fc(0.5)-2' --gpu 0 --version 10 --lambda_kd_shared 2.2 --lambda_kd_feature 2.2 --train_epoch 110 --total_epoch 50 --save_interval 60 --batch_size 64 --seed 3407
# done
# python  surf_cl_teacher.py --data_root '/remote-home/share/mengxichen/CASIA-SURF/' --dataset 'surf'  --method 'cl(neg-full)_cat-logp(0.1)_div-sum-64-drop-mean-23-7-110' --gpu 4 --version 2 --lambda_kd_shared 1.8 --lambda_kd_feature 1.8 --train_epoch 110 --total_epoch 50 --save_interval 60 --seed 3407
# python  surf_uncl_teacher_dif.py --data_root '/remote-home/share/mengxichen/CASIA-SURF/' --dataset 'surf'  --method 'uncl(pre-neg-full)_cat-logp(0.1)_div-sum-64-drop-mean-23-7-110' --gpu 0 --version 2 --lambda_kd_shared 1.8 --lambda_kd_feature 1.8 --train_epoch 60 --total_epoch 0 --save_interval 60 --seed 3407

# python  surf_cl_teacher.py --data_root '/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race/' --dataset 'cefa'  --method 'cl(neg-full)_cat-logp(0.1)_div-sum-64-drop-mean-23-7-110' --gpu 3 --version 1 --lambda_kd_shared 1.8 --lambda_kd_feature 1.8 --train_epoch 110 --total_epoch 50 --save_interval 60 --seed 3407
# python  surf_uncl_teacher_dif.py --data_root '/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race/' --dataset 'cefa'  --method 'uncl(pre-neg-full)_cat-logp(0.1)_div-sum-64-drop-mean-23-7-110' --gpu 7 --version 1 --lambda_kd_shared 1.8 --lambda_kd_feature 1.8 --train_epoch 60 --total_epoch 0 --save_interval 60 --seed 3407
