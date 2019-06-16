for var in "$@"
do
    nohup python scripts/validate.py aleppo.p --gpu=$var > validate_${var}.out &
    echo "Started build with gpu ${var}"
    sleep 20
done
