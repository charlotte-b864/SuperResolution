wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

mkdir train
mkdir train/high_res
mkdir train/low_res
mkdir train/raw

mkdir valid
mkdir valid/high_res
mkdir valid/low_res
mkdir valid/raw

mv DIV2K_train_HR/* train/raw/
mv DIV2K_valid_HR/* valid/raw/

python3 proc_imgs.py

rm DIV2K_train_HR.zip
rm DIV2K_valid_HR.zip

rm -rf DIV2K_train_HR/
rm -rf DIV2K_valid_HR/
