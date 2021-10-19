echo "Would you like to setup the virtualenv?(y/n)"
read setup_venv
if [ $setup_venv == 'y' ] || [ $setup_venv == 'Y' ] 
then
    . ./setup_virtualenv.sh
else
    source ~/dev/.venv/py3ros2/bin/activate
fi

echo "Installing vive_ros2 dependencies."
pip3 install -r requirements.txt
sudo apt-get install libsdl2-dev

echo "Would you like to install steam? (y/n)"
read install_steam
if [ $install_steam == 'y' ] || [ $install_steam == 'Y' ]
then
    sudo apt-get install steam
    echo "Please verify steam was installed correctly. Press enter to continue."
    read pass
    echo "Please run steam, update, login and install steamVR. Then press enter to continue."
    read pass
    cd ~/.steam/steam/steamapps/common/SteamVR/resources/settings/
    echo 'Please make the following changes to the default.vrsettings file, at path $cd ~/.steam/steam/steamapps/common/SteamVR/resources/settings/default.vrsettings - '
    echo "requiredHmd: false"
    echo "powerOffOnExit: false"
    echo "turnOffScreensTimeout: 500000"
    echo "turnOffControllersTimeout: 300000"
    echo "Press enter when ready to make changes."
    read pass
    vi ./default.vrsettings
fi
