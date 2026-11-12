"""Serve one loaded operation checkpoint on the GPU; never imports robot output."""
import argparse
import json
import signal

import zmq

from runtime import Predictor
from rpc import decode_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5560")
    args = parser.parse_args()
    predictor = Predictor(args.bundle, args.task, args.checkpoint, args.device)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, 9 * 1024 * 1024)
    socket.bind(args.bind)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps(dict(event="ready", task=args.task, checkpoint=predictor.weights, endpoint=args.bind)), flush=True)
    try:
        while not stopped:
            if not socket.poll(200):
                continue
            try:
                arrays, steps = decode_request(socket.recv_multipart())
                result = predictor.predict(*arrays, steps=steps)
                positions = result["positions"].astype("<f4")
                socket.send_multipart([json.dumps(dict(ok=True, task=args.task, shape=list(positions.shape),
                    inference_seconds=result["inference_seconds"], semantics=result["semantics"])).encode(), positions.tobytes()])
            except Exception as error:
                socket.send_multipart([json.dumps(dict(ok=False, error=f"{type(error).__name__}: {error}")).encode()])
    finally:
        socket.close()
        context.term()
        predictor.close()


if __name__ == "__main__":
    main()
