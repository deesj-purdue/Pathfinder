# Pathfinder

Pathfinder is a chest-mounted camera and sensor system that can detect obstacles for visually impaired people and provide haptic feedback for safety and path routing.


# Git repo setup

The `main` branch is for any code needing to be shared between the Jetson Nano and the ESP 32, such as communication API's.

The `esp` branch is for any code specific to the ESP 32.

The `jn` branch is for any code specific to the Jetson Nano.

Any sub-feature development branches should be prefixed with the branch you are working on, e.g. `esp-jackson-lidar`. Then, when searching for specific sub-branches, the command `git branch --list "esp*"` can be used to see only the branches for that component.

# LIDAR Setup

`lidar/api.py` expects a file called `lidar/_lidar.config` which contains the correct port to use the lidar. For example, on Windows it may contain `COM5`, but on the Jetson Nano it will have `/dev/ttyUSB0` 

# Setting up and pulling on Jetson Nano

**JN Requires `Python 3.12`**

Put this in the terminal to load my deploy key (read/write access to Pathfinder repo)

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/pathfinder-deploy-key
```


