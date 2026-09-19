#!/usr/bin/env python3
"""docker_bridge.py -- in-container half of the Docker rollout network bridge.

    python3 docker_bridge.py <port> <unix socket>

Listens on 127.0.0.1:<port> inside the container's otherwise network-less
namespace (`docker run --network none`) and pipes every connection to the
bind-mounted unix socket that run_rollouts.py's host-side `socat` forwards to
the SGLang server. The SWE-bench images ship no socat, so this is the mirror
of sandbox.sh's `socat TCP4-LISTEN ... UNIX-CONNECT`, in the stdlib the
images' base python has. Half-closes are propagated (write_eof) so streaming
responses are never cut short by a client that finished sending.
"""
import asyncio
import sys


async def _pump(src, dst):
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except Exception:
        pass
    finally:
        try:
            if dst.can_write_eof():
                dst.write_eof()
            else:
                dst.close()
        except Exception:
            pass


async def _handle(reader, writer, sock):
    try:
        ureader, uwriter = await asyncio.open_unix_connection(sock)
    except Exception as e:
        print(f"docker_bridge: connect {sock} failed: {e}", file=sys.stderr, flush=True)
        writer.close()
        return
    await asyncio.gather(_pump(reader, uwriter), _pump(ureader, writer))
    for w in (writer, uwriter):
        try:
            w.close()
        except Exception:
            pass


def main():
    port, sock = int(sys.argv[1]), sys.argv[2]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = loop.run_until_complete(asyncio.start_server(
        lambda r, w: _handle(r, w, sock), "127.0.0.1", port))
    try:
        loop.run_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
