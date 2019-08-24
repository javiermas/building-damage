CITIES=('aleppo' 'idlib' 'damascus' 'homs' 'hama' 'daraa' 'raqqa' 'deir-ez-zor')
for city in ${CITIES[@]}
do
    nohup python scripts/compute_features.py --city=$city --filename=${city}.p &
done
