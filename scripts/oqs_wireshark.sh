# Script for running easily the Open Quantum Safe Wireshark version
# Supports TLS 1.3 and advanced algorithm detection

xhost +si:localuser:root
xhost +si:localuser:$USER

sudo docker run --net=host --privileged --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" openquantumsafe/wireshark
