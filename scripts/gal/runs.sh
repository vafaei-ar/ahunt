
eval "$(conda shell.bash hook)"
conda activate gpu1

#script='fl_reg_all.py'
part=shared-gpu

for ntry in {0..20}; do 
for model in 'smp' 'vgg'; do

vsub -c "python grun.py --prefix 'res/' --model $model --ntry $ntry --epochs 10 --nqs 15 --nl 64" --part $part --name $model-$ntry --mem 70000 --time 12:00:00

done
done






