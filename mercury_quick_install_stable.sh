echo 'Quick install for Mercury, master branch'

git clone https://github.com/UoW-ATM/Mercury

cd Mercury

git submodule update --recursive --remote --init

deactivate

rmvirtualenv mercury-install-test

mkvirtualenv mercury-install-test

sudo apt-get install libproj-dev libgeos-dev build-essential python3-dev proj-data proj-bin

python -m pip install shapely cartopy --no-binary shapely --no-binary cartopy

pip install -r requirements.txt

wget https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1 -O ../mercury_public_dataset.zip

unzip ../mercury_public_dataset.zip -d ../input/

rm ../mercury_public_dataset.zip

./mercury.py -id -1 -cs -1