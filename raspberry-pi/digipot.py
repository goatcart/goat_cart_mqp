"""
The 'digipot' module creates an interface for the golf cart to interact with the
throttle potentiometer. There are 100 possible resistance values, which have
been labeled 0-99 in the module code.

The digipot should be connected to the Raspberry Pi as such:
    TODO insert ascii schematic here

For more info: https://datasheets.maximintegrated.com/en/ds/DS1804.pdf
"""

from time import sleep
import RPi.GPIO as GPIO

class Digipot:
    """The 'digipot' class exposes functions to permit throttle control."""
    def __init__(self, cs_pin, inc_pin, ud_pin):
        """Initialize the digipot for use."""
        # Set up class properties
        # Pins
        self.cs_pin = cs_pin
        self.inc_pin = inc_pin
        self.ud_pin = ud_pin
        # Characteristics
        self.min_wipe = 0
        self.max_wipe = 99
        self.resolution = 100 # ohm
        self.wiper_state = None # we'll know after setup

        # Set up all GPIO pins as output
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.cs_pin, GPIO.OUT)
        GPIO.setup(self.inc_pin, GPIO.OUT)
        GPIO.setup(self.ud_pin, GPIO.OUT)

        # Set the wiper_state to 0
        self.set_wipe(self.min_wipe)

    def set_wipe(self, wipe):
        """Set the digipot to a specific wipe value."""
        # Validate the attempted value
        if ~(self.min_wipe <= wipe <= self.max_wipe):
            return

    def set_resistance(self, resistance):
        """Set the digipot to a specific resistance."""
        return

    def set_mph(self, mph):
        """Set the digipot to a specific speed in mph."""
        return

    def _increase(self, num):
        """Increment the wiper 'num' times."""
        # Adjust the UD select to high (up)
        GPIO.output(self.ud_pin, GPIO.HIGH)
        # Enable the chip by asserting CS low
        GPIO.output(self.cs_pin, GPIO.LOW)
        # Pulse INC low n-1 times
        for _ in range(num-1):
            GPIO.output(self.inc_pin, GPIO.LOW)
            GPIO.output(self.inc_pin, GPIO.HIGH)
        # Assert INC low 1 additional time
        GPIO.output(self.inc_pin, GPIO.LOW)
        # Disable the chip by deasserting CS
        GPIO.output(self.cs_pin, GPIO.HIGH)
        # Deassert INC to complete the write
        GPIO.output(self.inc_pin, GPIO.HIGH)
        # Update current wiper state
        self.wiper_state += num
        if self.wiper_state > self.max_wipe:
            self.wiper_state = self.max_wipe

    def _decrease(self, num):
        """Decrement the wiper 'num' times."""
        # Adjust the UD select to low (down)
        GPIO.output(self.ud_pin, GPIO.LOW)
        # Enable the chip by asserting CS low
        GPIO.output(self.cs_pin, GPIO.LOW)
        # Pulse INC low num-1 times
        for _ in range(num-1):
            GPIO.output(self.inc_pin, GPIO.LOW)
            GPIO.output(self.inc_pin, GPIO.HIGH)
        # Assert INC low 1 additional time
        GPIO.output(self.inc_pin, GPIO.LOW)
        # Disable the chip by deasserting CS
        GPIO.output(self.cs_pin, GPIO.HIGH)
        # Deassert INC to complete the write
        GPIO.output(self.inc_pin, GPIO.HIGH)
        # Update current wiper state
        self.wiper_state -= num
        if self.wiper_state < 0:
            self.wiper_state = 0

    def _write_non_volatile(self, val):
        """Set the non-volatile register to 'val' wipes on boot."""
        # Ensure we're attempting to write a valid NV state
        if ~(self.min_wipe <= val <= self.max_wipe):
            return # Don't attempt anything if out-of-range
        # Decrease the wiper to 0 wipes
        self.set_wipe(0)
        # Adjust the UD select to high (up)
        GPIO.output(self.ud_pin, GPIO.LOW)
        # Enable the chip by asserting CS low
        GPIO.output(self.cs_pin, GPIO.LOW)
        # Pulse INC low 'val' times
        for _ in range(val):
            GPIO.output(self.inc_pin, GPIO.LOW)
            GPIO.output(self.inc_pin, GPIO.HIGH)
        # Disable the chip by de-asserting CS
        GPIO.output(self.cs_pin, GPIO.HIGH)
        # Wait >10ms for position to be written to NV register
        sleep(0.05) # 50 ms
        # Update current wiper state
        self.wiper_state = val
