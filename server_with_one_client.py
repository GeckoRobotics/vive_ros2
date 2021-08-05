# Copyright (c) 2021 Gecko Robotics, Inc. All rights reserved.

import argparse
import logging
from multiprocessing import Queue, Process, Pipe
from pathlib import Path
from threading import Thread, Event

from vive_tracker_client import ViveTrackerClient
from vive_server.vive_tracker_server import ViveTrackerServer
from vive_server.gui import GuiManager

def run_server(port: int, pipe: Pipe, logging_queue: Queue, config: Path, use_gui: bool, should_record: bool = False):
    vive_tracker_server = ViveTrackerServer(port=port, pipe=pipe, logging_queue=logging_queue, use_gui=use_gui,
                                            config_path=config, should_record=should_record)
    vive_tracker_server.run()


if __name__ == "__main__":

    # Below from server

    parser = argparse.ArgumentParser(description='Vive tracker server')
    parser.add_argument('--headless', default=False, help='if true will not run the gui')
    parser.add_argument('--port', default=8000, help='port to broadcast tracker data on')
    parser.add_argument('--ip', default="192.168.56.1", help='server ip')
    parser.add_argument('--config', default=f"~/vive_ros2/config.yml", # TODO make windows
                        help='tracker configuration file')
    args = parser.parse_args()

    logger_queue = Queue()
    gui_conn, server_conn = Pipe()
    config = Path(args.config).expanduser()
    string_formatter = logging.Formatter(fmt='%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt="%H:%M:%S")

    client = ViveTrackerClient(
                            host=args.ip,
                            port=args.port,
                            tracker_name="tracker_1",
                            should_record=False)

    client_message_queue = Queue()
    client_kill_thread = Event()
    client_thread = Thread(target=client.run_threaded, args=(client_message_queue, client_kill_thread,))
    # client_thread = Thread(target=client.run_threaded_log, args=(client_kill_thread,))
    client_thread.start()

    if args.headless:
        p = Process(target=run_server, args=(args.port, server_conn, logger_queue, config, False,))
        p.start()
        try:
            # This should be updated to be a bit cleaner
            while True:
                print(string_formatter.format(logger_queue.get()))
        finally:
            client_kill_thread.set()
            client_thread.join()
            p.kill()
    else:
        p = Process(target=run_server, args=(args.port, server_conn, logger_queue, config, True,))
        p.start()
        try:
            gui = GuiManager(gui_conn, logger_queue)
            gui.start()
        finally:
            client_kill_thread.set()
            client_thread.join()
            p.kill()

    ### Below from client
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", default=False, help="debug flag", type=str2bool)
    # parser.add_argument("--collect", default=False, help="debug flag", type=str2bool)
    # args = parser.parse_args()
    # logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    #                     datefmt="%H:%M:%S", level=logging.DEBUG if args.debug is True else logging.INFO)
    # HOST, PORT = "127.0.0.1", 8000
    # client = ViveTrackerClient(host=HOST, port=PORT, tracker_name="tracker_1", should_record=args.collect)
    # client.update()