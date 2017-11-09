## declare an array variable
declare -a arr=("CPI"	"GDP"	"UR"	"IR Policy Rate"	"LR10"	"LR10-IR"	"Exrate Euro for 1 USD")
declare -a arr1=(1 2 3 4 5)

## now loop through the above array
for i in "${arr[@]}"
do
	for j in "${arr1[@]}"
	do
	   python neuralnet.py --VAR "$i" --lags "$j"
	   # or do whatever with individual element of the array
	   pkill -9 python
	done
done

