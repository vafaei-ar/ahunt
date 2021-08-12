source vmini
eval "$(conda shell.bash hook)"
conda activate gpu1
export OMP_NUM_THREADS=4

# 'mnist' 'cifar10'
for fname in 'cifar10' ; do
# for plan in 1; do
for i in {0..99}; do

vsub -c "python gsc.py $fname $i" --part public-gpu --name $fname-p$plan-$i --mem 60000 --time 11:00:00 --ngpu 1
#vsub -c "python cifar_sc.py" --name anom2 --mem 60000 --time 11:00:00 --ngpu 1

done
# done
done





