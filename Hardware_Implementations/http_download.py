import network
import socket
import os
import time

SSID = ""
PASSWORD = ""

with open("secrets") as secrets:
    for line in secrets:
        secret = line.strip().split(":")
        if secret[0] == "SSID":
            print(f"Setting SSID: {secret[1]}")
            SSID = secret[1]
        if secret[0] == "PW":
            print(f"Setting PW: {secret[1]}")
            PASSWORD = secret[1]

def connect_to_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    print(f"Connecting to {SSID}")
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        print(".", end="")
        time.sleep(0.5)
    print("Connected", wlan.ifconfig())
    return wlan

def disconnect_wifi(wlan: network.WLAN):
    if wlan.isconnected():
        wlan.disconnect()
        print("Disconnected from Wifi")


def start_server():
    # global sck
    addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
    sck = socket.socket()
    sck.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sck.bind(addr)
    sck.listen(1)
    print(f"Server is listening on http://{addr[0]}")

    serve = True
    while serve:
        # action = input("Serve again? (y/n)\n")
        # if action.lower() == "y":
        client, addr = sck.accept()
        #     print("Client connected from", addr)


        try:
            request = client.recv(1024).decode()
            print("Reques:", request)

            if "GET /download/" in request:
                start = request.find("/download/") + len("/download/")
                end = request.find(" ", start)
                filename = request[start:end]

                if filename in os.listdir():
                    file_size = os.stat(filename)[6]
                    client.send("HTTP/1.1 200 OK\r\n")
                    client.send("Content-Type: application/octet-stream\r\n")
                    client.send(f"Content-Length: {file_size}\r\n")
                    client.send(f"Content-Disposition: attachment; filename={filename}\r\n")
                    client.send("Connection: close\r\n\r\n")

                    with open(filename, "rb") as file:
                        while chunk := file.read(512):
                            client.send(chunk)
                    print(f"Sent file: {filename}")

                else:
                    client.send("HTTP/1.1 404 Not Found\r\n\r\nFile not found.")
            
            else:
                # files = os.listdir("*.raw")
                client.send("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
                client.send(b"<h1>Pico Fingerprint API</h1>")
                # client.send(b"<ul>")
                # for file in files:
                #     payload = f"<li>{file}</li>"
                #     client.send(payload.encode())
                # client.send("</ul>")
                client.send(b"<p>Use <code>/download/&lt;filename&gt;</code> to get a file.</p>")

        except Exception as e:
            print("Error: ", e)
        finally:
            client.close()
    # else:
    #     serve = False
    sck.close()
    time.sleep(1)
    print("Socket closed")

try:
    wlan = connect_to_wifi()
    start_server()
except Exception as e:
    print(e)
finally:
    disconnect_wifi(wlan)