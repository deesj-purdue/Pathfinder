import time
import sys
import argparse

#!/usr/bin/env python3
# pin_test.py - Blink an LED on a Jetson Nano GPIO pin (1s on / 1s off)
# Usage: python pin_test.py PIN [--mode BOARD|BCM]

try:
    import Jetson.GPIO as GPIO
except Exception as e:
    print("Failed to import Jetson.GPIO:", e, file=sys.stderr)
    sys.exit(1)

def main():
    p = argparse.ArgumentParser(description="Blink an LED on a Jetson Nano GPIO pin (1s on/off).")
    p.add_argument("pin", type=int, help="GPIO pin number (BOARD or BCM based on --mode).")
    p.add_argument("--mode", choices=("BOARD", "BCM"), default="BOARD", help="Pin numbering mode (default: BOARD).")
    args = p.parse_args()

    mode = GPIO.BOARD if args.mode == "BOARD" else GPIO.BCM
    GPIO.setmode(mode)
    pin = args.pin

    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    try:
        print(f"Blinking pin {pin} ({args.mode}) — Ctrl+C to stop")
        while True:
            GPIO.output(pin, GPIO.HIGH)
            print("LED ON")
            time.sleep(1.0)
            GPIO.output(pin, GPIO.LOW)
            print("LED OFF")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()