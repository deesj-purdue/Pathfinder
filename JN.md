# Jetson Nano branch


# LIDAR Setup

`lidar/api.py` expects a file called `lidar/_lidar.config` which contains the correct port to use the lidar. For example, on Windows it may contain `COM5`, but on the Jetson Nano it will have `/dev/ttyUSB0`


# Setting up and pulling on Jetson Nano

Put this in the terminal to load my deploy key (read/write access to Pathfinder repo)

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/pathfinder-raspberry-pi-deploy-key
```
