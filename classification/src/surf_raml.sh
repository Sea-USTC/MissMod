for i in `seq 0.4 0.2 2`;
do

    python  surf_raml.py --data_root '/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race'  --method 'raml' --gpu 3 --version 1 --weight_uni 1.8 --weight_inf $i

done

