echo "Would you like to setup the virtualenv?(y/n)"
read setup_venv

if [ $setup_venv == 'y' ] || [ $setup_venv == 'Y' ] 
then
    . ./setup_virtualenv.sh
else
    source /usr/local/py3ros2/bin/setup.activate
fi

echo "Installing vive_ros2 dependencies."

pip3 install -r requirements.txt
sudo apt-get install libsdl2-dev