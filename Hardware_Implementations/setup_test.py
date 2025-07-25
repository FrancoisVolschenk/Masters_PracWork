from pyfingerprint import *
from machine import UART, Pin

opt = UART(1)
opt.init(57600, bits=8, parity=None, stop=1, rx=Pin(5), tx=Pin(4))

capt = UART(0)
capt.init(57600, bits=8, parity=None, stop=1, rx=Pin(13), tx=Pin(12))

optical = PyFingerprint(opt)
capacitive = PyFingerprint(capt)

### Verify both sensors work
halt = False
if not optical.verifyPassword():
    print("The optical sensor cannot be reached")
    halt = True
else:
    print("Optical online")
if not capacitive.verifyPassword():
    print("The capacitive sensor could not be reached")
    halt = True
else:
    print("Capacitive online")
if not halt:
    print("Both sensors operational")