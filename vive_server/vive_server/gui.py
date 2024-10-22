from abc import ABC, abstractmethod
import queue
from pathlib import Path
import math
import numpy as np

import dearpygui.dearpygui as dpg
import logging

from models import Configuration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RED = [255, 0, 0, 255]
PURPLE = [128, 0, 128, 255]
GREEN = [0, 255, 0, 255]
BLUE = [0, 0, 255, 255]
GREY = [128, 128, 128, 255]
GRIDLINES = [128, 128, 128, 50]
BLACK = [0, 0, 0, 255]

TRACKER_COLOR = [0, 255, 255, 255]
REFERENCE_COLOR = [255, 0, 255, 255]
CONTROLLER_COLOR = [255, 255, 255, 255]


class Page(ABC):
    def __init__(self, name: str, gui_manager):
        self.name = name
        self.gui_manager = gui_manager

    def show(self) -> bool:
        if not dpg.does_item_exist(self.name):
            with dpg.window(label=self.name, tag=self.name, autosize=True, on_close=self.clear):
                pass  # Window contents will be added in subclasses
            return True
        return False

    @abstractmethod
    def update(self, system_state: dict):
        pass

    def clear(self, sender, data):
        dpg.delete_item(self.name)


# render 3d scene from the top down (size of dot represent the scale on the z)
# Moving in and out changes the x and y axis by changing the virtual camera configuration
class Scene:
    def __init__(self, width=1000, height=500, name="scene"):
        self.logger = logging.getLogger(__name__ + ".Scene")
        self.name = name
        self.width = width
        self.height = height
        self.scale_x, self.scale_y = self.width / 10, self.width / 10
        self.z_scale, self.z_offset = 10, 1.0

        self.center = [self.width / 2, self.height / 2]
        self.bottom_left = [self.width, self.height]

        self.is_dragging = False
        self.last_mouse_pos = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.translation = [0, 0]

    def add(self):
        with dpg.drawlist(width=self.width, height=self.height, tag=self.name):
            dpg.draw_rectangle(pmin=(0, 0), pmax=self.bottom_left, color=(0, 0, 0, 255), fill=(0, 0, 0, 255), tag="background")
        
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(callback=self.on_drag)
            dpg.add_mouse_release_handler(callback=self.on_release)

    def on_drag(self, sender, app_data, user_data):
        if not self.is_dragging:
            self.is_dragging = True
            self.last_mouse_pos = app_data[1:]
        else:
            delta_x = app_data[1] - self.last_mouse_pos[0]
            delta_y = app_data[2] - self.last_mouse_pos[1]
            
            self.rotation_y += delta_x * 0.01
            self.rotation_x += delta_y * 0.01
            
            self.last_mouse_pos = app_data[1:]

    def on_release(self, sender, app_data, user_data):
        self.is_dragging = False

    def transform_point(self, point):
        # Apply rotation
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(self.rotation_x), -np.sin(self.rotation_x)],
                          [0, np.sin(self.rotation_x), np.cos(self.rotation_x)]])
        
        rot_y = np.array([[np.cos(self.rotation_y), 0, np.sin(self.rotation_y)],
                          [0, 1, 0],
                          [-np.sin(self.rotation_y), 0, np.cos(self.rotation_y)]])
        
        rotation = np.dot(rot_y, rot_x)
        rotated_point = np.dot(rotation, point)
        
        # Apply translation and scaling
        transformed_point = [
            (rotated_point[0] + self.translation[0]) * self.scale_x + self.center[0],
            (rotated_point[1] + self.translation[1]) * self.scale_y + self.center[1],
            rotated_point[2] * self.z_scale
        ]
        
        return transformed_point

    def draw(self, device_state):
        dpg.delete_item(self.name, children_only=True)
        dpg.draw_rectangle(parent=self.name, pmin=(0, 0), pmax=self.bottom_left, color=(0, 0, 0, 255), fill=(0, 0, 0, 255), tag="background")
        self.add_axes()
        self.draw_scales()
        
        for device, tracker_msg in device_state.items():
            if 'tracker' in device or 'controller' in device:
                if tracker_msg is not None:
                    self.draw_tracker(tracker_msg)

    def add_axes(self):
        length = 100  # You can adjust this value to change the length of all axes
        origin = self.transform_point([0, 0, 0])
        x_end = self.transform_point([length / self.scale_x, 0, 0])
        y_end = self.transform_point([0, length / self.scale_y, 0])
        z_end = self.transform_point([0, 0, length / self.z_scale])

        dpg.draw_line(parent=self.name, p1=origin[:2], p2=x_end[:2], color=(255, 0, 0, 255), thickness=2, tag="axis_x")
        dpg.draw_line(parent=self.name, p1=origin[:2], p2=y_end[:2], color=(0, 255, 0, 255), thickness=2, tag="axis_y")
        dpg.draw_line(parent=self.name, p1=origin[:2], p2=z_end[:2], color=(0, 0, 255, 255), thickness=2, tag="axis_z")
        
        # Add axis labels
        dpg.draw_text(parent=self.name, pos=x_end[:2], text="X", color=(255, 0, 0, 255), size=20, tag="label_x")
        dpg.draw_text(parent=self.name, pos=y_end[:2], text="Y", color=(0, 255, 0, 255), size=20, tag="label_y")
        dpg.draw_text(parent=self.name, pos=z_end[:2], text="Z", color=(0, 0, 255, 255), size=20, tag="label_z")

        dpg.draw_circle(parent=self.name, center=origin[:2], radius=4, color=(255, 255, 255, 255), fill=(255, 255, 255, 255), tag="axis_origin")

    def draw_tracker(self, tracker_msg):
        if tracker_msg is None:
            return

        try:
            point = self.transform_point([tracker_msg.x, tracker_msg.y, tracker_msg.z])
            diameter = abs(point[2]) + self.z_offset * self.z_scale
            dpg.draw_text(parent=self.name, pos=[point[0], point[1] - diameter - 15], text=f'{tracker_msg.device_name}', color=TRACKER_COLOR, size=13, tag=f"{tracker_msg.device_name}txt")
            dpg.draw_circle(parent=self.name, center=point[:2], radius=diameter, color=TRACKER_COLOR, fill=TRACKER_COLOR, tag=f"{tracker_msg.device_name}dot")
            _, _, yaw = tracker_msg.rotation_as_scipy_transform().as_euler("xyz")
            radius = 20 + diameter / 2
            pt2 = [point[0] - radius * math.cos(yaw), point[1] + radius * math.sin(yaw)]
            dpg.draw_line(parent=self.name, p1=point[:2], p2=pt2, color=PURPLE, thickness=3, tag=f"{tracker_msg.device_name}line")
        except AttributeError as e:
            self.logger.error(f"Error drawing tracker {tracker_msg.device_name if hasattr(tracker_msg, 'device_name') else 'Unknown'}: {e}")

    def real_pose_from_pixels(self, point):
        return [(point[0] - self.center[0]) / self.scale_x, (point[1] - self.center[1]) / self.scale_y]

    def real_pose_to_pixels(self, point):
        # Apply rotation to the point
        rotated_x = point[0] * math.cos(self.rotation_y) - point[1] * math.sin(self.rotation_y)
        rotated_y = point[0] * math.sin(self.rotation_y) + point[1] * math.cos(self.rotation_y)
        rotated_y = rotated_y * math.cos(self.rotation_x) - point[2] * math.sin(self.rotation_x)
        rotated_z = rotated_y * math.sin(self.rotation_x) + point[2] * math.cos(self.rotation_x)

        return [(rotated_x * self.scale_x + self.center[0]), 
                (rotated_y * self.scale_y + self.center[1]),
                rotated_z]

    def draw_scales(self):
        tick_h = 5
        for x in range(0, self.width, 50):
            dpg.draw_line(parent=self.name, p1=[x, self.height], p2=[x, 0], color=GRIDLINES, thickness=1, tag=f"{x}xgridline")
            dpg.draw_line(parent=self.name, p1=[x, self.height], p2=[x, self.height - tick_h], color=GREY, thickness=1, tag=f"{x}xtick")
            x_real = self.real_pose_from_pixels([x, 0])[0]
            dpg.draw_text(parent=self.name, pos=[x, self.height - tick_h - 20], text=f'{round(x_real, 1)}m', color=GREY, size=13, tag=f"{x}xticktext")
        for y in range(0, self.height, 50):
            dpg.draw_line(parent=self.name, p1=[0, y], p2=[self.width, y], color=GRIDLINES, thickness=1, tag=f"{y}ygridline")
            dpg.draw_line(parent=self.name, p1=[0, y], p2=[tick_h, y], color=GREY, thickness=1, tag=f"{y}ytick")
            y_real = self.real_pose_from_pixels([0, y])[1]
            dpg.draw_text(parent=self.name, pos=[tick_h + 5, y - 2], text=f'{round(y_real, 1)}m', color=GREY, size=13, tag=f"{y}yticktext")


class DevicesPage(Page):
    def __init__(self, gui_manager, name="devices"):
        super().__init__(name, gui_manager)
        self.devices_shown = []

    def update(self, system_state):
        for device in system_state:
            serial = system_state[device].serial_num
            if device not in self.devices_shown:
                self.devices_shown.append(device)
                dpg.add_input_text(f"{device}:{serial}##name", default_value=system_state[device].device_name,
                                   on_enter=True, callback=self.update_device_name,
                                   callback_data=(device, serial))
                dpg.add_text(f"{serial}_txt", color=GREY)
            else:
                dpg.set_value(f"{serial}_txt", f"x: {round(system_state[device].x, 2)}, "
                                           f"y: {round(system_state[device].y, 2)}, "
                                           f"z: {round(system_state[device].z, 2)}")

    def update_device_name(self, sender, data):
        device, serial = data
        new_name = dpg.get_value(f"{device}:{serial}##name")
        config = self.gui_manager.get_config()
        config.name_mappings[serial] = new_name
        self.gui_manager.update_config(config)

    def clear(self, sender, data):
        super(DevicesPage, self).clear(sender, data)
        self.devices_shown = []


# Calibration page includes scene with special configuration
class CalibrationPage(Page):
    def __init__(self, name: str, gui_manager):
        super(CalibrationPage, self).__init__(name, gui_manager)
        self.trackers = []
        self.origin_tracker = None
        self.pos_x_tracker = None
        self.pos_y_tracker = None

    def show(self):
        if super(CalibrationPage, self).show():
            with dpg.window(self.name):
                dpg.add_text("instructions##calibration", default_value="Please select a tracker for "
                                                                    "each axis. Available trackers "
                                                                    "are listed below for convenience:")
                dpg.add_spacing()
                dpg.add_text("trackers##calibration", default_value=str(self.trackers))
                dpg.add_input_text(f"origin##calibration", default_value="", callback=self.update_origin)
                dpg.add_input_text(f"+x##calibration", default_value="", callback=self.update_pos_x)
                dpg.add_input_text(f"+y##calibration", default_value="", callback=self.update_pos_y)
                dpg.add_button("Start calibration", callback=self.run_calibration)

    def update_origin(self, sender, data):
        self.origin_tracker = dpg.get_value("origin##calibration")

    def update_pos_x(self, sender, data):
        self.pos_x_tracker = dpg.get_value("+x##calibration")

    def update_pos_y(self, sender, data):
        self.pos_y_tracker = dpg.get_value("+y##calibration")

    def run_calibration(self, sender, data):
        # verify valid input (trackers + unique)
        if self.origin_tracker in self.trackers and \
                self.pos_y_tracker in self.trackers and \
                self.pos_x_tracker in self.trackers and \
                self.origin_tracker != self.pos_x_tracker and \
                self.origin_tracker != self.pos_y_tracker and \
                self.pos_x_tracker != self.pos_y_tracker:
            self.gui_manager.call_calibration(self.origin_tracker, self.pos_x_tracker, self.pos_y_tracker)
        else:
            logger.warning("Invalid tracker entered for calibration")

    def update(self, system_state: dict):
        trackers = []
        for key in system_state:
            if "tracker" in key:
                trackers.append(system_state[key].device_name)
        if len(trackers) > len(self.trackers):
            self.trackers = trackers
            dpg.set_value("trackers##calibration", str(trackers))

    def clear(self, sender, data):
        super(CalibrationPage, self).clear(sender, data)
        self.trackers = []


class TestCalibrationPage:
    def __init__(self):
        pass


class ConfigurationPage(Page):
    def show(self):
        super(ConfigurationPage, self).show()

    def update(self, system_state):
        config = self.gui_manager.get_config()
        if config is not None:
            config_dict = dict(self.gui_manager.get_config())
            for value in config_dict:
                if not dpg.does_item_exist(f"{value}##config"):
                    dpg.add_input_text(f"{value}##config", default_value=str(config_dict[value]),
                                   on_enter=True, callback=self.update_config_entry,
                                   callback_data=value)
                else:
                    dpg.set_value(f"{value}##config", str(config_dict[value]))

    def update_config_entry(self, sender, data):
        config = self.gui_manager.get_config()


class CustomHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class VisualizationPage:
    def __init__(self, gui_manager):
        self.gui_manager = gui_manager
        self.scene = Scene(name="main_scene")
        self.devices_page = DevicesPage(name="Devices List", gui_manager=self.gui_manager)
        self.configuration_page = ConfigurationPage(name="Configuration", gui_manager=self.gui_manager)
        self.calibration_page = CalibrationPage(name="Calibration", gui_manager=self.gui_manager)
        self.log_queue = queue.Queue()
        self.log_handler = CustomHandler(self.log_queue)
        logger.addHandler(self.log_handler)
        self.log_window_created = False

    def show(self):
        dpg.add_button(label="Save Configuration", callback=self.save_config)
        dpg.add_same_line()
        dpg.add_button(label="Refresh", callback=self.refresh)
        dpg.add_same_line()
        dpg.add_button(label="Calibrate", callback=self.calibrate)
        dpg.add_same_line()
        dpg.add_button(label="Test Calibration", callback=self.test_calibration)
        dpg.add_same_line()
        dpg.add_button(label="List Devices", callback=self.list_devices)
        dpg.add_same_line()
        dpg.add_button(label="Show Configuration", callback=self.show_configuration)
        dpg.add_same_line()
        dpg.add_button(label="Logs", callback=self.logs)
        self.scene.add()

    def save_config(self, sender, data):
        self.gui_manager.save_config()

    def refresh(self, sender, data):
        self.gui_manager.refresh_system()

    def calibrate(self, sender, data):
        self.calibration_page.show()

    def test_calibration(self, sender, data):
        pass

    def list_devices(self, sender, data):
        self.devices_page.show()

    def show_configuration(self, sender, data):
        self.configuration_page.show()

    def logs(self, sender, app_data, user_data):
        if not self.log_window_created:
            with dpg.window(label="Logger", tag="logger_window"):
                dpg.add_text("Log output will appear here")
                dpg.add_text("", tag="log_output")
            self.log_window_created = True
        else:
            dpg.configure_item("logger_window", show=True)

    def update_logs(self):
        if not self.log_window_created:
            return

        log_text = ""
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                log_text += self.log_handler.format(record) + "\n"
            except queue.Empty:
                break

        if log_text and dpg.does_item_exist("log_output"):
            current_text = dpg.get_value("log_output")
            dpg.set_value("log_output", current_text + log_text)

    def update(self, system_state: dict):
        self.scene.draw(system_state)
        if dpg.does_item_exist("Devices List"):
            self.devices_page.update(system_state)
        if dpg.does_item_exist("Configuration"):
            self.configuration_page.update(system_state)
        if dpg.does_item_exist("Calibration"):
            self.calibration_page.update(system_state)
        self.update_logs()

    def clear(self):
        pass


class GuiManager:
    def __init__(self, pipe, logging_queue):
        self._pipe = pipe
        self._logging_queue = logging_queue
        self._server_config = None
        self._page = VisualizationPage(self)

    def on_render(self, sender, data):
        while self._logging_queue.qsize() > 0:
            try:
                record = self._logging_queue.get_nowait()
                logger.handle(record)
            except queue.Empty:
                pass

        system_state = {}
        while self._pipe.poll():
            data = self._pipe.recv()
            if "state" in data:
                system_state = data["state"]
            if "config" in data:
                self._server_config = data["config"]
        self._page.update(system_state)

    def update_config(self, config):
        self._server_config = config
        self._pipe.send({"config": self._server_config})

    def get_config(self) -> Configuration:
        if self._server_config is not None:
            return self._server_config.copy()

    def save_config(self, path: Path = None):
        self._pipe.send({"save": path})

    def refresh_system(self):
        self._pipe.send({"refresh": None})

    def call_calibration(self, origin, pos_x, pos_y):
        self._pipe.send({"calibrate": (origin, pos_x, pos_y)})

    # Will Run the main gui
    def start(self):
        dpg.create_context()
        with dpg.window(label="Vive Server", tag="main_window"):
            self._page.show()

        dpg.create_viewport(title="Vive Server", width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        while dpg.is_dearpygui_running():
            self.on_render(None, None)
            dpg.render_dearpygui_frame()

        dpg.destroy_context()




























