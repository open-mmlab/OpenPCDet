import pickle
import socket
from logging import Logger
from typing import Any, Optional

# re-name the method below, to maintain consistency with RadarProcessor
from pcdet.utils.common_utils import create_logger as get_logger


class Streamer():
    """
    A class for sending data through a socket connection.

    This class can be used either as a client or a server, and is intended to
    be used for both sides of a streaming connection.

    A server will wait for connections from a client, while the client will
    attempt to connect to the server, and both can send and receive messages
    over their connection once it's established.

    In general, both server and client should follow a pattern like this:

        streamer = Streamer(mode=YOUR_MODE)
        streamer.handshake()
        with streamer.get_socket() as sock:
            with streamer.get_connection(sock) as connection:
                ... send and receive messages here ...

    See the handshake() method for an example.

    NOTE: In all situations, the server must be started first! Otherwise, the
          client's attempt to connect will immediately fail.
    """
    def __init__(
            self,
            mode: str = 'server',
            host: str = 'localhost',
            port: int = 6000,
            log: Optional[Logger] = None,
        ):
        assert mode in ['server', 'client'], 'Invalid mode! Must be "server" or "client"'
        self.mode = mode
        self.header_size = 10
        self.header_encoding = "utf-8"
        self.packet_size = 4096
        self.socket_address = (host, port)
        self.socket_params = (socket.AF_INET, socket.SOCK_STREAM)
        if log is None:
            self.log = get_logger()
        else:
            self.log = log

    def get_socket(self) -> socket.SocketType:
        """Create a socket object, ready to use."""
        return socket.socket(*self.socket_params)

    def get_connection(self, sock: socket.SocketKind) -> socket.SocketType:
        """
        Set up a connection based on the mode.

        Note: for any client/server pair, the server's connection must be
              created first, or the client will fail to connect.

        Args:
            sock: a socket object
        Returns:
            an actively-connected socket object
        """
        if self.mode == 'server':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(self.socket_address)
            sock.listen()
            connection, _ = sock.accept()
        elif self.mode == 'client':
            sock.connect(self.socket_address)
            connection = sock
        return connection

    def handshake(self):
        """Perform a handshake to establish a connection"""
        self.log.debug("Performing handshake")
        with self.get_socket() as sock:
            with self.get_connection(sock) as connection:
                if self.mode == 'client':
                    self._send_message(b"Ready?", connection)
                else:
                    response = self._receive_message(connection)
                    if response != b"Ready?":
                        raise Exception("Handshake failed")

                if self.mode == 'server':
                    self._send_message(b"Yep!", connection)
                else:
                    response = self._receive_message(connection)
                    if response != b"Yep!":
                        raise Exception("Handshake failed")

    def _send_message(self, msg: bytes, connection: socket.SocketType):
        self.log.debug("Sending message")
        msg_length = len(msg)
        header = bytes(f"{msg_length:<{self.header_size}}", self.header_encoding)
        connection.sendmsg((header, msg))
        self.log.debug("... message sent")

    def _receive_message(self, connection: socket.SocketType) -> bytes:
        self.log.debug("Waiting for message")
        msg_length = int(
            connection.recv(self.header_size).decode(self.header_encoding)
        )
        msg = b''
        while len(msg) < msg_length:
            packet = connection.recv(self.packet_size)
            if not packet:
                raise Exception("Incomplete message!")
            msg += packet
        self.log.debug("... message received")
        return msg

    def receive_object(self, connection:socket.SocketType) -> Any:
        """Read a pickle-able object from an open connection"""
        msg = self._receive_message(connection)
        object = pickle.loads(msg)
        return object

    def send_object(self, object: Any, connection: socket.SocketType):
        """Send a pickle-able object through an open connection"""
        msg = pickle.dumps(object)
        self._send_message(msg, connection)
