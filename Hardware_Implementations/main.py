from pyfingerprint import *
from machine import UART, Pin
from time import sleep

def convert_to_characteristics(fp, passCount = 0):
    CHAR_BUFFER = [FINGERPRINT_CHARBUFFER1, FINGERPRINT_CHARBUFFER2]
    PASS_STRING = ["first", "scond"]
    result = fp.convertImage(charBufferNumber=CHAR_BUFFER[passCount])

    if result:
        print(f"Converted {PASS_STRING[passCount]} pass to characteristics")
    else:
        print(f"Could not create {PASS_STRING[passCount]} pass characteristics")
    return result

def scan_finger(fp):
    fp.ledOn(colour=FINGERPRINT_LED_PURPLE, control=FINGERPRINT_LED_BREATHING, flashSpeed=0x7D, flashCount=0x00)
    while not fp.readImage():
        print("Reading scanner...")
        sleep(0.5)
    print("Finger captured!")
    fp.ledOff()

def store_template(fp, pos = 0):
    index = fp.storeTemplate(positionNumber=pos, charBufferNumber=FINGERPRINT_CHARBUFFER1)
    print(f"Stored the captured finger at index {index}")

def match_finger(fp):
    (pos, score) = fp.searchTemplate(charBufferNumber=FINGERPRINT_CHARBUFFER1, positionStart=0, count=-1)
    print(f"Found match at {pos} with a score of {score}")
    if score > 0:
        fp.ledOn(colour=FINGERPRINT_LED_BLUE, control=FINGERPRINT_LED_FLASHING, flashSpeed=0x7D, flashCount=0x00)
    else:
        fp.ledOn(colour=FINGERPRINT_LED_RED, control=FINGERPRINT_LED_FLASHING, flashSpeed=0x7D, flashCount=0x00)
    sleep(2)
    fp.ledOff()

def capture_image(fp, path = "fingerprint.raw"):
    print(f"Downloading image to {path}")
    optical.downloadImage(path)
    print("Done")

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

    # Indicate startup sequence completed
    capacitive.ledOn(colour=FINGERPRINT_LED_BLUE, control=FINGERPRINT_LED_CONTINUOUS, flashSpeed=0x7D, flashCount=0x00)
    sleep(2)
    capacitive.ledOff()

    try:
        while not halt:
            sensor_select = input("1: Optical\n2: Capacitive\n3: QUIT\n:")
            if sensor_select not in ["1", "2"]:
                halt = True
            else:
                sensor_type = optical if sensor_select == "1" else capacitive
                action = input("1: Enrol new finger\n2: Match Finger\n3: Capture Image\n:")

            if not halt and action == "3":
                scan_finger(sensor_type)
                capture_image(sensor_type, "fingerprint.raw")

            if not halt:
                scan_finger(sensor_type)
                success = convert_to_characteristics(sensor_type, 0)
                halt = not success
                sleep(0.5)
            if not halt:
                scan_finger(sensor_type)
                success = convert_to_characteristics(sensor_type, 1)
                halt = not success
            if not halt:
                if sensor_type.createTemplate():
                    print("Created template")
                else:
                    print("Could not create template")
                    halt = True

            if not halt:
                if action == "1": # Enrol
                    if not halt:
                        #TODO: Check for number of available positions, number of used positions, recommend a position. 
                        #TODO: Check input number for range
                        pos = int(input("At which index would you like to store the new sample?\n:"))
                        store_template(sensor_type, pos)
                if action == "2": #Match
                    match_finger(sensor_type)
                    #TODO: Up the threshold for matching to have some resonable confidence metric
    except Exception as e:
        print("Something went wrong")
        print(e)



# if not halt:
#     characteristics = sensor_type.downloadCharacteristics(charBufferNumber=FINGERPRINT_CHARBUFFER1)
#     print(characteristics)

# sleep(1)
