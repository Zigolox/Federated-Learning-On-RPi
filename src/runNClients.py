#!/usr/bin/env python3
import asyncio
import json

start_server = "python server.py"


start_client = "python client.py --server_address=127.0.0.1:8080"


async def run(cmd):

    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, _ = await proc.communicate()
    if "client" in cmd:
        return json.loads(stdout.decode())
    return


async def testNClients(num_client: int, num_epoch: int = 3, num_rounds=3):
    clients = [
        run(start_client + f" --cid={i} --epoch {num_epoch} --nb_clients {num_client}")
        for i in range(num_client)
    ]
    server = run(start_server + f" --num_rounds {num_rounds}")
    client_data = await asyncio.gather(server, *clients)
    return client_data[1:]


def wrapNClients(*args, **kwargs):
    asyncio.run(testNClients(*args, **kwargs))


if __name__ == "__main__":
    wrapNClients(num_client=2)
