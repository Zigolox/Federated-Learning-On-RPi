#!/usr/bin/env python3
from subprocess import Popen


commands = [
    [
        "python",
        "client.py",
        "--cid=0",
        "--server_address=127.0.0.1:8080",
    ],
    [
        "python",
        "server.py",
    ],
]
procs = [Popen(i) for i in commands]
for p in procs:
    p.wait()
