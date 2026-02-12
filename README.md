# Pathfinder

Pathfinder is a chest-mounted camera and sensor system that can detect obstacles for visually impaired people and provide haptic feedback for safety and path routing.


# Git repo setup

The `main` branch is for any code needing to be shared between the Raspberry Pi 5 and the ESP32, such as communication API's.

The `esp/*` branch prefix is for any code specific to the ESP32. E.g., `esp/main` is the main branch for the ESP32.

The `rp/*` branch prefix is for any code specific to the RP5. E.g., `rp/main` is the main branch for the RP5.

An example branch would be `rp/jackson-lidar`, which we could then merge into `rp/main` when it is done developing, it should NOT be merged into `main`. 

When searching for specific branch prefixes, the command `git branch --list "esp/*"` can be used to see only the branches for that component.

