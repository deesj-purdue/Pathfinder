# Pathfinder


Pathfinder is a chest-mounted camera and sensor system that can detect obstacles for visually impaired people and provide haptic feedback for safety and path routing.


# Git repo setup

The `main` branch is for any code needing to be shared between the Jetson Nano and the ESP 32, such as communication API's.

The `esp` branch is for any code specific to the ESP 32.

The `jn` branch is for any code specific to the Jetson Nano.

Any sub-feature development branches should be prefixed with the branch you are working on, e.g. `esp-jackson-lidar`. Then, when searching for specific sub-branches, the command `git branch --list "esp*"` can be used to see only the branches for that component.
