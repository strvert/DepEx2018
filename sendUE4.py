from __future__ import print_function
import socket
import time
from contextlib import closing


def sendue4():
    host = '127.0.0.1'
    port = 4000
    bufsize = 4096

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with closing(sock):
        sock.connect((host, port))
        sock.send(b'Hello world')
        # print(sock.recv(bufsize))


if __name__ == '__main__':
    sendue4()