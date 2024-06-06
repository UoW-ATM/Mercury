echo 'Quick install and test run for Mercury, dev branch\n\n'

git clone -b dev https://github.com/UoW-ATM/Mercury

cd Mercury

git submodule update --recursive --remote --init

# Source the virtualenvwrapper script
source /usr/local/bin/virtualenvwrapper.sh  # Update this path to where your virtualenvwrapper.sh is located

# Deactivate any active virtual environment
deactivate 2>/dev/null || true

rmvirtualenv mercury-install-test

mkvirtualenv mercury-install-test

sudo apt-get install libproj-dev libgeos-dev build-essential python3-dev proj-data proj-bin

python -m pip install shapely cartopy --no-binary shapely --no-binary cartopy

pip install -r requirements.txt

wget https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1 -O ../mercury_public_dataset.zip

unzip ../mercury_public_dataset.zip -d ../input/

rm ../mercury_public_dataset.zip

./mercury.py -id -1 -cs -1

echo '\n\nInstallation and test run went well. Test virtual environment is mercury-install-test'
