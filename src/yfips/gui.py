"""Live 2D visualizer: listens to the UDP detection stream and plots
each robot's current (x, y, yaw) on the world plane."""

import json
import math
import socket
import threading
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from yfips import config


class UDPListener(threading.Thread):
    def __init__(self, host, port, timeout_s=1.0):
        super().__init__(daemon=True)
        self.timeout_s = timeout_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.state = {}  # id -> (x, y, yaw, t)
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                m = json.loads(data.decode("utf-8"))
                rid = int(m["id"])
                with self.lock:
                    self.state[rid] = (float(m["x"]), float(m["y"]),
                                        float(m["yaw"]), float(m.get("t", time.time())))
            except (ValueError, KeyError):
                continue

    def snapshot(self):
        now = time.time()
        with self.lock:
            return {rid: v for rid, v in self.state.items()
                    if (now - v[3]) <= self.timeout_s}


def main():
    cfg = config.load()
    udp = cfg["udp"]
    listener = UDPListener(udp["host"], int(udp["port"]))
    listener.start()

    wc = cfg.get("world_corners_m", [[0, 0], [5, 0], [5, 5], [0, 5]])
    xs = [p[0] for p in wc]
    ys = [p[1] for p in wc]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("YF-IPS live")
    ax.plot(xs + [xs[0]], ys + [ys[0]], "k--", lw=1)

    arrow_len = 0.25

    try:
        while plt.fignum_exists(fig.number):
            snap = listener.snapshot()
            for p in list(ax.patches):
                p.remove()
            for t in list(ax.texts):
                t.remove()
            for rid, (x, y, yaw, _) in snap.items():
                dx, dy = arrow_len * math.cos(yaw), arrow_len * math.sin(yaw)
                ax.add_patch(mpatches.FancyArrow(
                    x, y, dx, dy, width=0.04, head_width=0.12,
                    length_includes_head=True, color="C0"))
                ax.text(x + 0.1, y + 0.1, str(rid), fontsize=9)
            plt.pause(0.1)
    finally:
        listener.running = False


if __name__ == "__main__":
    main()
