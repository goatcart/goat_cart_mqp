#!/usr/bin/env python3

"""
This script allows for diagnostic user tests of the throttle's digital
potentiometer.
"""

import sys
from digipot import Digipot

# The chosen GPIO pins go here
# Pick the proper pins at some point.
CS_PIN, INC_PIN, UD_PIN = 1, 1, 1

def main():
    """Prompt for user input and exit on quit."""
    # Instantiate the digipot
    digipot = Digipot(CS_PIN, INC_PIN, UD_PIN)
    # Print a welcome message
    print("Enter values below to set digipot behavior.")
    print("Format: <value> <type>")
    print("Examples:")
    print("\t50\tSet the potentiometer to the 50th wipe (from 0 to 99)")
    print("\t800ohm\tSet the potentiometer to 1200 ohms")
    print("\t5.5mph\tSet the potentiometer to a speed of 5.5 mph")
    print("\treset\tReboot the potentiometer and revert to wipe 0")
    print("\tquit\tExit the program")
    # Enter prompt loop
    while True:
        parse(digipot, input("> "))
    return

def parse(digipot, user_str):
    """Parse the user input and perform the corresponding instruction."""
    if user_str == "reset": # RESET
        return
    elif user_str == "quit": # QUIT
        sys.exit()
    elif user_str.isdigit(): # WIPE
        digipot.set_wipe(int(user_str))
    elif user_str[-3:] == "ohm": # OHM
        value = int(user_str.split()[0])
        digipot.set_resistance(value)
    elif user_str[-3:] == "mph": # MPH
        value = float(user_str.split()[0])
        digipot.set_mph(value)
    return

if __name__ == "__main__":
    main()
