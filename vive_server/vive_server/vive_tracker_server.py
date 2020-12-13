"""
OpenVr based Vive tracker server
"""

import argparse
import json
import logging
import logging.handlers
import socket
from multiprocessing import Queue, Process, Pipe
from os.path import expanduser
from pathlib import Path
from pprint import pprint
from typing import List
from typing import Optional

from base_server import Server
from gui import MainGui
from models import ViveDynamicObjectMessage, ViveStaticObjectMessage
from triad_openvr import TriadOpenVR


def construct_socket_msg(data: ViveDynamicObjectMessage) -> str:
    """
    Send vive tracker message to socket

    Args:
        data: ViveTracker Message to send

    Returns:
        message in string to send

    """
    json_data = json.dumps(data.json(), sort_keys=False)
    json_data = "&" + json_data
    json_data = json_data + "\r"  # * (512 - len(json_data))
    return json_data


class ViveTrackerServer(Server):
    """
    Defines a UDP vive tracker server that constantly "shout out" messages at (HOST, PORT)

    Utilizes OpenVR as its interaction with SteamVR. For hardware setup, please see this tutorial:
    http://help.triadsemi.com/en/articles/836917-steamvr-tracking-without-an-hmd

    """

    def __init__(self, port, pipe, logging_queue, use_gui=False, buffer_length: int = 1024, should_record: bool = False,
                 output_file_path: Path = Path(expanduser("~") + "/vive_ros2/data/RFS_track.txt")):
        """
        Initialize socket and OpenVR
        
        Args:
            port: desired port to open
            logging_queue: handler with where to send logs
            buffer_length: maximum buffer (tracker_name) that it can listen to at once
            should_record: should record data or not
            output_file_path: output file's path
        """
        super(ViveTrackerServer, self).__init__(port)
        self.logger = logging.getLogger("ViveTrackerServer")
        self.logger.addHandler(logging.handlers.QueueHandler(logging_queue))
        self.logger.setLevel(logging.INFO)
        self.pipe = pipe
        self.use_gui = use_gui

        self.socket = self.initialize_socket()
        self.triad_openvr: Optional[TriadOpenVR] = self.reconnect_triad_vr(debug=False)
        self.should_record = should_record
        self.output_file_path = output_file_path
        self.output_file = None
        if not self.output_file_path.exists():
            self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_file_path.open('w')
        self.buffer_length = buffer_length

    def run(self):
        """
        Initialize a server that runs forever.

        This server can be put into a multi-process module to run concurrently with other servers.

        This server will listen for client's request for a specific tracker's name

        It will compute that tracker's information

        It will then send that information
        Returns:
            None
        """
        self.logger.info(f"Starting server at {self.ip}:{self.port}")
        self.logger.info("Connected VR devices: \n###########\n" + str(self.triad_openvr) + "###########")
        while True:
            messages = {}
            # Transmit data over the network
            try:
                tracker_name, addr = self.socket.recvfrom(self.buffer_length)
                tracker_name = tracker_name.decode()
                if tracker_name in self.get_tracker_names():
                    message = self.poll_tracker(tracker_name=tracker_name)
                    messages[tracker_name] = message
                    if message is not None:
                        socket_message = construct_socket_msg(data=message)
                        self.socket.sendto(socket_message.encode(), addr)
                        if self.should_record:
                            self.record(data=message)
                else:
                    self.logger.error(f"Tracker [{tracker_name}] not found")
            except socket.timeout:
                self.logger.info("Did not receive connection from client")
            except Exception as e:
                self.logger.error(e)

            # Update the GUI
            if self.use_gui:
                for tracker_name in self.get_tracker_names():
                    # Make sure all trackers are shown in the GUI regardless of if they are being subscribed to
                    if tracker_name not in messages:
                        message = self.poll_tracker(tracker_name=tracker_name)
                        messages[tracker_name] = message
                for reference_name in self.get_tracking_reference_names():
                    if reference_name not in messages:
                        message = self.poll_tracking_reference(reference_name)
                        messages[reference_name] = message

                self.pipe.send(messages)

    def poll_tracker(self, tracker_name) -> Optional[ViveDynamicObjectMessage]:
        """
        Polls tracker message by name

        Note:
            Server will attempt to reconnect if tracker name is not found.

        Args:
            tracker_name: the vive tracker message intended to poll

        Returns:
            ViveTrackerMessage if tracker is found, None otherwise.
        """
        tracker = self.get_device(device_name=tracker_name)
        if tracker is not None:
            message: Optional[ViveDynamicObjectMessage] = self.create_dynamic_message(device=tracker,
                                                                                      device_name=tracker_name)
            return message
        else:
            self.reconnect_triad_vr()
        return None

    def poll_controller(self, controller_name) -> Optional[ViveDynamicObjectMessage]:
        """
        Polls controller message by name

        Note:
            Server will attempt to reconnect if tracker name is not found.

        Args:
            controller_name: the vive tracker message intended to poll

        Returns:
            ViveTrackerMessage if tracker is found, None otherwise.
        """
        controller = self.get_device(device_name=controller_name)
        if controller is not None:
            message: Optional[ViveDynamicObjectMessage] = self.create_dynamic_message(device=controller,
                                                                                      device_name=controller_name)
            return message
        else:
            self.reconnect_triad_vr()
        return None

    def poll_tracking_reference(self, tracking_reference_name) -> Optional[ViveStaticObjectMessage]:
        """
        Polls tracking reference message by name

        Note:
            Server will attempt to reconnect if tracker name is not found.

        Args:
            tracking_reference_name: the vive tracking reference intended to poll

        Returns:
            ViveTrackerMessage if tracker is found, None otherwise.
        """
        tracking_reference = self.get_device(device_name=tracking_reference_name)
        if tracking_reference is not None:
            message: Optional[ViveStaticObjectMessage] = self.create_static_message(device=tracking_reference,
                                                                                    device_name=tracking_reference_name)
            return message
        else:
            self.reconnect_triad_vr()
        return None

    def get_device(self, device_name):
        """
        Given tracker name, find the tracker instance

        Args:
            device_name: desired tracker's name to find

        Returns:
            tracker instance if found, None otherwise
        """
        return self.triad_openvr.devices.get(device_name, None)

    def create_dynamic_message(self, device, device_name) -> Optional[ViveDynamicObjectMessage]:
        """
        Create dynamic object message given device and device name

        Note:
            it will attempt to reconnect to OpenVR if conversion or polling from device went wrong.

        Args:
            device: tracker instance
            device_name: the device's name corresponding to this tracker

        Returns:
            Vive dynamic message if this is a successful conversion, None otherwise

        """
        try:
            _, _, _, r, p, y = device.get_pose_euler()
            x, y, z, qw, qx, qy, qz = device.get_pose_quaternion()
            vel_x, vel_y, vel_z = device.get_velocity()
            p, q, r = device.get_angular_velocity()

            # Rotate velocity in local frame
            # vel = [0, vel_x, vel_y, vel_z]
            # quat = [qw, qx, qy, qz]
            # _, vel_x, vel_y, vel_z = q_mult(quat, q_mult(vel, q_conjugate(quat)))

            message = ViveDynamicObjectMessage(valid=True, x=x, y=y, z=z,
                                               qx=qx, qy=qy, qz=qz, qw=qw,
                                               vel_x=vel_x, vel_y=vel_y, vel_z=vel_z,
                                               p=p, q=q, r=r,
                                               device_name=device_name)
            return message
        except OSError as e:
            print(f"OSError: {e}. Need to restart Vive Tracker Server")
            self.reconnect_triad_vr()
        except Exception as e:
            print(f"Exception {e} has occurred, this may be because device {device} "
                  f"is either offline or malfunctioned")
            self.reconnect_triad_vr()
            return None

    def create_static_message(self, device, device_name) -> Optional[ViveStaticObjectMessage]:
        """
        Create tracker message given device and device name

        Note:
            it will attempt to reconnect to OpenVR if conversion or polling from tracker went wrong.

        Args:
            device: device instance
            device_name: the device's name corresponding to this tracker

        Returns:
            Vive static message if this is a successful conversion, None otherwise

        """
        try:
            x, y, z, qw, qx, qy, qz = device.get_pose_quaternion()
            message = ViveStaticObjectMessage(valid=True, x=x, y=y, z=z,
                                              qx=qx, qy=qy, qz=qz, qw=qw,
                                              device_name=device_name)
            return message
        except OSError as e:
            print(f"OSError: {e}. Need to restart Vive Tracker Server")
            self.reconnect_triad_vr()
        except Exception as e:
            print(f"Exception {e} has occurred, this may be because device {device} "
                  f"is either offline or malfunctioned")
            self.reconnect_triad_vr()
            return None

    def reconnect_triad_vr(self, debug=False):
        """
        Attempt to reconnect to TriadOpenVR

        Notes:
            this method will automatically assign self.triad_openvr

        Args:
            debug: **deprecated flag

        Returns:
            openvr instance
        """
        openvr = TriadOpenVR()

        if debug:
            self.logger.debug(
                f"Trying to reconnect to OpenVR to refresh devices. "
                f"Devices online:")
            pprint(openvr.devices)
        self.triad_openvr = openvr
        return openvr

    def get_tracker_names(self) -> List[str]:
        """
        Get a list of trackers

        Returns:
            list of tracker names

        """
        return self.get_device_list(filters=["tracker"])

    def get_tracking_reference_names(self) -> List[str]:
        """
        Get a list of tracking references (base stations)

        Returns:
            list of references names

        """
        return self.get_device_list(filters=["reference"])

    def get_controller_names(self) -> List[str]:
        """
        Get a list of controllers

        Returns:
            list of controller names

        """
        return self.get_device_list(filters=["controller"])

    def get_device_list(self, filters=None) -> List[str]:
        result = []
        for device_name in self.triad_openvr.devices.keys():
            if filters is None:
                result.append(device_name)
            else:
                for s in filters:
                    if s in device_name:
                        result.append(device_name)
        return result

    def record(self, data: ViveDynamicObjectMessage):
        """
        Record the current data

        Args:
            data: current ViveTrackerMessage to record

        Returns:
            None
        """
        x, y, z, qw, qx, qy, qz = data.x, data.y, data.z, data.qw, data.qx, data.qy, data.qz
        recording_data = f"{x}, {y},{z},{qw},{qx},{qy},{qz}"
        m = f"Recording: {recording_data}"
        self.logger.info(m)
        self.output_file.write(recording_data + "\n")


def run_server(port, pipe, logging_queue, use_gui, should_record=False):
    vive_tracker_server = ViveTrackerServer(port=port, pipe=pipe, logging_queue=logging_queue, use_gui=use_gui,
                                            should_record=should_record)
    vive_tracker_server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vive tracker server')
    parser.add_argument('--headless', default=False, help='if true will not run the gui')
    parser.add_argument('--port', default=8000, help='port to broadcast tracker data on')
    args = parser.parse_args()

    logger_queue = Queue()
    gui_conn, server_conn = Pipe()
    string_formatter = logging.Formatter(fmt='%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt="%H:%M:%S")

    if args.headless:
        p = Process(target=run_server, args=(args.port, server_conn, logger_queue, False,))
        p.start()
        try:
            # This should be updated to be a bit cleaner
            while True:
                print(string_formatter.format(logger_queue.get()))
        finally:
            p.kill()
    else:
        p = Process(target=run_server, args=(args.port, server_conn, logger_queue, True,))
        p.start()
        try:
            gui = MainGui(gui_conn, logger_queue)
            gui.start()
        finally:
            p.kill()