# Copyright (c) 2021 Gecko Robotics, Inc. All rights reserved.

import argparse
import logging
from multiprocessing import Queue, Process, Pipe
import os
from pathlib import Path
import socket
import time
from threading import Thread, Event, enumerate

from vive_tracker_client import ViveTrackerClient
from vive_server.vive_tracker_server import ViveTrackerServer
from vive_server.gui import GuiManager

def run_server(port: int, pipe: Pipe, logging_queue: Queue, config: Path, use_gui: bool, should_record: bool = False):
    vive_tracker_server = ViveTrackerServer(port=port, pipe=pipe, logging_queue=logging_queue, use_gui=use_gui,
                                            config_path=config, should_record=should_record)
    vive_tracker_server.run()


if __name__ == "__main__":

    # .venv\Scripts\activate & python server_with_one_client.py

    parser = argparse.ArgumentParser(description='Vive tracker server')
    parser.add_argument('--port', default=8000, help='port to broadcast tracker data on')
    # parser.add_argument('--ip', default="192.168.56.1", help='server ip')
    parser.add_argument('--ip', default=str(socket.gethostbyname(socket.gethostname())), help='server ip')
    parser.add_argument('--config', default=Path(os.getcwd()) / Path("config"),
                        help='tracker configuration file')
    args = parser.parse_args()

    logger_queue = Queue()
    gui_conn, server_conn = Pipe()
    config = Path(args.config).expanduser()
    string_formatter = logging.Formatter(fmt='[%(asctime)s][%(name)s][%(levelname)s][%(message)s', datefmt="%H:%M:%S")

    p = Process(target=run_server, args=(args.port, server_conn, logger_queue, config, True,),daemon=False)
    
    client = ViveTrackerClient(
                            host=args.ip,
                            port=args.port,
                            tracker_name="tracker_1", # TODO DRY
                            should_record=False)

    client_message_queue = Queue()
    client_message_queue.cancel_join_thread()
    client_kill_thread = Event()
    client_thread = Thread(target=client.run_threaded, args=(client_message_queue, client_kill_thread,), daemon=False)
    # client_thread.start()

    try:
        client_thread.start()
        p.start()
        gui = GuiManager(gui_conn, logger_queue)
        gui.start()
        
    finally:
        client_kill_thread.set()
        client_thread.join()
        p.kill()
        # p.terminate()