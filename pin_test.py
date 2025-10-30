import sys
import time
import argparse

#!/usr/bin/env python3
# pin_test.py - simple Jetson Nano GPIO LED tester
# Usage:
#   sudo python3 pin_test.py PIN [--bcm] [--blink N] [--interval SEC] [--duration SEC]
# Examples:
#   sudo python3 pin_test.py 12           # set BOARD pin 12 HIGH for 5s then LOW
#   sudo python3 pin_test.py 18 --bcm     # set BCM pin 18 HIGH for 5s then LOW
#   sudo python3 pin_test.py 12 --blink 10 --interval 0.3


try:
    import Jetson.GPIO as GPIO
except Exception as e:
    print("Error importing Jetson.GPIO:", e)
    print("On a Jetson Nano, install python3 -m pip install Jetson.GPIO and run with sudo.")
    sys.exit(1)

def main():
    p = argparse.ArgumentParser(description="Jetson Nano GPIO LED test")
    p.add_argument("pin", type=int, help="GPIO pin number (BOARD or BCM depending on --bcm)")
    p.add_argument("--bcm", action="store_true", help="use BCM numbering (default: BOARD)")
    p.add_argument("--blink", type=int, default=0, help="number of blink cycles (on+off). If 0, just turn on for duration.")
    p.add_argument("--interval", type=float, default=0.5, help="blink interval in seconds (default 0.5)")
    p.add_argument("--duration", type=float, default=5.0, help="how long to keep LED on if not blinking (seconds)")
    args = p.parse_args()

    mode = GPIO.BCM if args.bcm else GPIO.BOARD
    GPIO.setmode(mode)
    GPIO.setwarnings(False)
    pin = args.pin

    try:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        if args.blink > 0:
            for i in range(args.blink):
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(args.interval)
                GPIO.output(pin, GPIO.LOW)
                time.sleep(args.interval)
        else:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(args.duration)
            GPIO.output(pin, GPIO.LOW)
        print("Done. Cleaning up GPIO.")
    except KeyboardInterrupt:
        print("Interrupted by user. Cleaning up GPIO.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()