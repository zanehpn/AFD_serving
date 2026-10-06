"""Prevent port-probe false failures while rejecting live listeners."""
import socket
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_worker import check_ports


def config_for(port):
    # Other probes bind ephemeral ports, leaving the target as the only fixed port.
    return dict(api_port=port, afd_port=0, dp_rpc_port=port+10)


def test_live_listener_rejected_with_port():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        port = server.getsockname()[1]
        server.listen()
        with pytest.raises(OSError, match=f'127.0.0.1:{port}'):
            check_ports(config_for(port))


def test_closed_reusable_server_is_not_live_owner():
    with socket.socket() as server, socket.socket() as client:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        port = server.getsockname()[1]
        server.listen()
        client.connect(('127.0.0.1', port))
        peer, _ = server.accept()
        peer.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b''
        client.close()
        peer.close()
    check_ports(config_for(port))
