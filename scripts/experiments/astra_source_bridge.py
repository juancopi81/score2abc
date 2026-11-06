"""Fixed Unix-socket bridge for one sandboxed local MCP source server."""

from __future__ import annotations

import socket
import sys
import threading


def main() -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(sys.argv[1])

        def send() -> None:
            try:
                for line in sys.stdin.buffer:
                    connection.sendall(line)
                connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        threading.Thread(target=send, daemon=True).start()
        with connection.makefile("rb") as stream:
            for line in stream:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
