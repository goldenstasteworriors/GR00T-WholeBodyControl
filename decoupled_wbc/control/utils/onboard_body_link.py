"""Single-session body command transport; a broken connection never reconnects."""

import socket
import struct
import threading

import msgpack


def send_packet(sock, message):
    payload = msgpack.packb(message, use_bin_type=True)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_packet(sock):
    def read_exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size - len(chunks))
            if not chunk:
                raise EOFError("body connection closed")
            chunks.extend(chunk)
        return chunks

    size = struct.unpack("!I", read_exact(4))[0]
    return msgpack.unpackb(read_exact(size), raw=False)


class OnboardBodyLink:
    def __init__(self, host, port=5560):
        self.socket = socket.create_connection((host, port), timeout=2.0)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.settimeout(None)
        self.lock = threading.Lock()
        self.received = threading.Condition()
        self.sequence = 0
        self.acknowledged = 0
        self.disconnected = False
        self.context = None
        self.confirmed = threading.Event()
        self.status = {}
        threading.Thread(target=self._receive, daemon=True).start()

    def _receive(self):
        try:
            while True:
                status = receive_packet(self.socket)
                with self.received:
                    self.status = status
                    self.acknowledged = int(status.get("sequence", 0))
                    if status.get("latched"):
                        self.disconnected = True
                    if status.get("damping_confirmed"):
                        self.confirmed.set()
                    self.received.notify_all()
        except (OSError, EOFError):
            self.disconnected = True

    def publish(self, low_cmd, damping=False):
        if self.disconnected and not damping:
            return False
        packet = {
            "kind": "damping" if damping else "position",
            "mode_machine": int(low_cmd.mode_machine),
            "mode_pr": int(low_cmd.mode_pr),
            "motors": [[int(m.mode), m.q, m.dq, m.tau, m.kp, m.kd]
                       for m in low_cmd.motor_cmd[:29]],
            "context": self.context,
        }
        try:
            with self.lock:
                self.sequence += 1
                packet["sequence"] = self.sequence
                send_packet(self.socket, packet)
                with self.received:
                    if not self.received.wait_for(lambda: self.acknowledged >= self.sequence, timeout=0.2):
                        self.disconnected = True
                        return False
                if self.disconnected and not damping:
                    return False
            return True
        except OSError:
            self.disconnected = True
            return False

    def close(self):
        # Explicit process exit is different from an unexplained network loss.
        self.confirmed.clear()
        try:
            with self.lock:
                send_packet(self.socket, {"kind": "exit"})
            if not self.confirmed.wait(2.0):
                raise RuntimeError("Robot did not confirm damping; onboard hold may still be active")
        finally:
            self.socket.close()


class OnboardBodyPublisher:
    """Publisher interface used by the existing body sender, never by hands."""

    def __init__(self, host):
        self.link = OnboardBodyLink(host)

    def Write(self, command):
        damping = all(m.kp == 0.0 and m.kd == 8.0 for m in command.motor_cmd[:29])
        return self.link.publish(command, damping=damping)

    def Close(self):
        self.link.close()

