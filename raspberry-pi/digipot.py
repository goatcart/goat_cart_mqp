"""
The 'digipot' module creates an interface for the golf cart to interact with the
throttle potentiometer. There are 100 possible resistance values, which have
been labeled 0-99 in the module code.

Three functions are exposed for intended use:
    - set_wipe()
    - set_resistance()
    - set_mph()

The digipot should be connected to the Raspberry Pi as below. Note that P##
refers to the physical pin layout on the RPi, while BCM## refers to the
Broadcomm pin numbers. The BCM numbering must be used in the module code this
script provides.

 Raspberry Pi        DS1804 Pot
+-------------+     +-----------+
|             |     |           |
| P16/BCM23 [---------] P1 ~INC |
|             |     |           |
| P18/BCM24 [---------] P2 U/~D |
|             |     |           |
| P24/BCM8  [---------] P7 ~CS  |
|             |     |           |
+-------------+     +-----------+

For more info: https://datasheets.maximintegrated.com/en/ds/DS1804.pdf
"""

from time import sleep
import RPi.GPIO as GPIO # pylint: disable=import-error

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
        self.max_wipe = 99
        self.max_resistance = 10000
        self.max_mph = 15
        self.wiper_state = None # we'll know after setup

        # Set up all GPIO pins as output
        GPIO.setmode(GPIO.BCM) # IMPORTANT: BCM means use Broadcomm numbering
        GPIO.setup(self.cs_pin, GPIO.OUT)
        GPIO.setup(self.inc_pin, GPIO.OUT)
        GPIO.setup(self.ud_pin, GPIO.OUT)

        # Set the wiper_state to 0
        self.set_wipe(0)

    def set_wipe(self, wipe):
        """Set the digipot to a specific wipe value."""
        # Validate the desired value
        if ~(0 <= wipe <= self.max_wipe):
            return
        if wipe > self.wiper_state: # We need to increase wipe
            self._increase(wipe - self.wiper_state)
        elif wipe < self.wiper_state: # We need to decrease wipe
            self._decrease(self.wiper_state - wipe)
        else: # We're already at the desired value
            return

    def set_resistance(self, resistance):
        """Set the digipot to a specific resistance."""
        # Validate the desired value
        if ~(0 <= resistance <= self.max_resistance):
            return
        # Round to the nearest wipe (multiple of 100 ohms)
        wipe_equivalent = int(round(resistance, -2)) / 100
        # Set the digipot
        self.set_wipe(wipe_equivalent)

    def set_mph(self, mph):
        """Set the digipot to a specific speed in mph."""
        # Validate the desired value
        if ~(0 <= mph <= self.max_mph):
            return
        # Calculate R/mph, given that max speed occurs at 5.5kohm
        resistance_per_mph = 5500.0 / self.max_mph
        # Find the nearest appropriate wipe
        wipe_equivalent = int(mph * resistance_per_mph)
        # Set the digipot
        self.set_wipe(wipe_equivalent)

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
        """Set the non-volatile register to 'val' wipes on boot.

        You shouldn't ever need to use this, but it's written for completeness.
        """
        # Ensure we're attempting to write a valid NV state
        if ~(0 <= val <= self.max_wipe):
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
