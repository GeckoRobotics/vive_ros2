sudo apt-get install python3-pip
sudo apt-get install python3-virtualenv
virtualenv --version
virtualenv -p /usr/bin/python3 ~/dev/.venv/py3ros2
source ~/dev/.venv/py3ros2/bin/activate
echo "Pleas verify the virtial environment was setup correctly. Press enter to continue."
read pass