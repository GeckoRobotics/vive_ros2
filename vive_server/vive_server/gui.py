from abc import ABC, abstractmethod
import queue
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation

import dearpygui.dearpygui as dpg
import logging
import time
import datetime

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

TRACKER_COLOR = (255, 215, 0, 255)  # Gold color for the crown
REFERENCE_COLOR = [255, 0, 255, 255]
CONTROLLER_COLOR = [255, 255, 255, 255]


class Page(ABC):
    def __init__(self, name: str, gui_manager):
        self.name = name
        self.gui_manager = gui_manager

    def show(self) -> bool:
        if not dpg.does_item_exist(self.name):
            with dpg.window(label=self.name, tag=self.name, autosize=False, on_close=self.clear):
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
        self.z_scale, self.z_offset = self.width / 10, 0

        self.center = [self.width / 2, self.height / 2]
        self.bottom_left = [self.width, self.height]

        self.is_dragging = False
        self.last_mouse_pos = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.translation = [0, 0]
        self.camera_rotation = Rotation.from_quat([0, 0, 0, 1])  # Identity quaternion

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

    def draw_crown(self, center, size, color, rotation):
        # Base of the crown (triangle)
        points = []
        num_points = 3
        base_radius = size * 0.8  # Slightly smaller than the spike length
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points + rotation + math.pi  # Start from the bottom
            x = center[0] + base_radius * math.cos(angle)
            y = center[1] + base_radius * math.sin(angle)
            points.append([x, y])
        
        dpg.draw_polygon(parent=self.name, points=points, color=color, fill=color)

        # Spikes of the crown
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points + rotation + math.pi  # Start from the bottom
            base_x = center[0] + base_radius * math.cos(angle)
            base_y = center[1] + base_radius * math.sin(angle)
            tip_x = center[0] + size * math.cos(angle)
            tip_y = center[1] + size * math.sin(angle)
            dpg.draw_triangle(parent=self.name, 
                              p1=[base_x, base_y],
                              p2=[tip_x, tip_y],
                              p3=[center[0] + base_radius * math.cos(angle + 2*math.pi/num_points), 
                                  center[1] + base_radius * math.sin(angle + 2*math.pi/num_points)],
                              color=color, fill=color)

        # Center jewel
        dpg.draw_circle(parent=self.name, center=center, radius=size/4, color=(255,0,0,255), fill=(255,0,0,255))

    def draw_tracker_axes(self, center, size, rotation_matrix):
        # Colors for each axis
        colors = [(255, 0, 0, 255),  # X-axis: Red
                  (0, 255, 0, 255),  # Y-axis: Green
                  (0, 0, 255, 255)]  # Z-axis: Blue

        # Draw axes
        for i in range(3):
            axis = rotation_matrix[:2, i]  # Get the first two elements of the i-th column
            end_point = center + axis * size
            dpg.draw_arrow(parent=self.name, p1=center, p2=end_point, color=colors[i], thickness=2, size=5)

        # Draw labels
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            axis = rotation_matrix[:2, i]  # Get the first two elements of the i-th column
            label_pos = center + axis * size * 1.1  # Place label slightly beyond arrow tip
            dpg.draw_text(parent=self.name, pos=label_pos, text=labels[i], color=colors[i], size=12)

    def draw_tracker(self, tracker_msg):
        if tracker_msg is None:
            return

        try:
            point = self.transform_point([tracker_msg.x, tracker_msg.y, tracker_msg.z])
            size = (abs(point[2]) + self.z_offset * self.z_scale) * 0.5  # Adjust size as needed

            # Get rotation matrix
            rotation = tracker_msg.rotation_as_scipy_transform()
            rotation_matrix = rotation.as_matrix()[:3, :3]  # Get the 3x3 rotation matrix

            # Draw the axes
            self.draw_tracker_axes(np.array(point[:2]), size, rotation_matrix)
            
            # Draw the label
            dpg.draw_text(parent=self.name, pos=[point[0], point[1] - size * 1.2], 
                          text=f'{tracker_msg.device_name}', color=(255, 255, 255, 255), size=13, 
                          tag=f"{tracker_msg.device_name}txt")

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

    def update(self, system_state):
        # Get the rotation matrix from the quaternion
        rotation_matrix = self.camera_rotation.as_matrix()

        for device, state in system_state.items():
            # Get the original position
            original_position = np.array([state.x, state.y, state.z])
            
            # Apply rotation to the position
            rotated_position = rotation_matrix @ original_position
            
            # Update the tracker's position
            self.trackers[device].position = rotated_position.tolist()
            
            # Rotate the tracker's orientation
            original_orientation = Rotation.from_quat([state.qx, state.qy, state.qz, state.qw])
            rotated_orientation = self.camera_rotation * original_orientation
            qx, qy, qz, qw = rotated_orientation.as_quat()
            
            # Update the tracker's orientation
            self.trackers[device].quaternion = [qw, qx, qy, qz]
            # ... update other properties as needed ...

    def set_camera_rotation(self, qx, qy, qz, qw):
        self.camera_rotation = Rotation.from_quat([qx, qy, qz, qw])

    def rotate_camera(self, axis, angle):
        """Rotate the camera around a given axis by the specified angle (in radians)."""
        rotation = Rotation.from_rotvec(axis * angle)
        self.camera_rotation = rotation * self.camera_rotation


class DevicesPage(Page):
    def __init__(self, name, gui_manager):
        super().__init__(name, gui_manager)
        self.window_tag = f"{name}_window"
        self.selected_device = None
        self.device_list = []
        self.is_recording = False

    def show(self):
        if not dpg.does_item_exist(self.window_tag):
            with dpg.window(label="Devices", tag=self.window_tag, width=400, height=200):
                with dpg.group(horizontal=True):
                    dpg.add_combo(tag="device_selector", callback=self.on_device_selected)
                    dpg.add_button(label="Start Recording", tag="record_button", callback=self.toggle_recording)
                dpg.add_separator()
                dpg.add_group(tag="device_info_group")
            
            # Set initial device selection
            self.refresh_devices()
        
        dpg.show_item(self.window_tag)

    def update(self, system_state):
        if not dpg.does_item_exist(self.window_tag):
            self.show()

        # Update device list if it has changed
        current_devices = set(system_state.keys())
        if current_devices != set(self.device_list):
            self.refresh_devices(system_state)

        # Update information for the selected device
        if self.selected_device and self.selected_device in system_state:
            self.update_device_info(self.selected_device, system_state[self.selected_device])

    def refresh_devices(self, system_state=None):
        if system_state is None or not isinstance(system_state, dict):
            system_state = {}
            if hasattr(self.gui_manager, '_server_config') and self.gui_manager._server_config is not None:
                if hasattr(self.gui_manager._server_config, 'trackers'):
                    system_state.update(self.gui_manager._server_config.trackers)
                if hasattr(self.gui_manager._server_config, 'tracking_references'):
                    system_state.update(self.gui_manager._server_config.tracking_references)
            else:
                self.gui_manager.add_log("Server configuration not available yet", level=logging.WARNING)
        
        self.device_list = list(system_state.keys())
        if dpg.does_item_exist("device_selector"):
            dpg.configure_item("device_selector", items=self.device_list)
        
        if self.selected_device not in self.device_list:
            if self.device_list:
                self.selected_device = self.device_list[0]
                if dpg.does_item_exist("device_selector"):
                    dpg.set_value("device_selector", self.selected_device)
            else:
                self.selected_device = None
                if dpg.does_item_exist("device_selector"):
                    dpg.set_value("device_selector", "")

    def on_device_selected(self, sender, app_data, user_data):
        self.selected_device = app_data
        dpg.delete_item("device_info_group", children_only=True)

    def update_device_info(self, device, state):
        dpg.delete_item("device_info_group", children_only=True)
        with dpg.group(parent="device_info_group"):
            dpg.add_text(f"{device}:")
            dpg.add_text(f"Position: x: {state.x:.4f}, y: {state.y:.4f}, z: {state.z:.4f}")
            dpg.add_text(f"Rotation: roll: {state.roll:.2f}, pitch: {state.pitch:.2f}, yaw: {state.yaw:.2f}")
            if hasattr(state, 'vel_x'):
                dpg.add_text(f"Velocity: x: {state.vel_x:.4f}, y: {state.vel_y:.4f}, z: {state.vel_z:.4f}")
            else:
                dpg.add_text("Velocity: N/A")

    def toggle_recording(self, sender, app_data, user_data):
        if self.selected_device:
            self.is_recording = not self.is_recording
            message = {"record": self.is_recording, "device": self.selected_device}
            self.gui_manager._pipe.send(message)
            
            if self.is_recording:
                dpg.configure_item("record_button", label="Stop Recording")
                self.gui_manager.add_log(f"Started recording data for device: {self.selected_device}")
            else:
                dpg.configure_item("record_button", label="Start Recording")
                self.gui_manager.add_log(f"Stopped recording data for device: {self.selected_device}")

    def clear(self):
        if dpg.does_item_exist(self.window_tag):
            dpg.delete_item(self.window_tag)
        self.selected_device = None
        self.device_list.clear()

    def hide(self):
        if dpg.does_item_exist(self.window_tag):
            dpg.hide_item(self.window_tag)

class VisualizationPage:
    def __init__(self, gui_manager):
        self.gui_manager = gui_manager
        self.scene = Scene(name="main_scene")
        self.devices_page = DevicesPage(name="Devices", gui_manager=self.gui_manager)

    def show(self):
        with dpg.window(label="Main Window", tag="main_window", width=1000, height=1080, pos=[400, 0]):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh", callback=self.refresh)
            self.scene.add()
        
        self.devices_page.show()
        self.show_logs()

    def save_config(self, sender, data):
        self.gui_manager.save_config()

    def refresh(self, sender, data):
        self.gui_manager.refresh_system()

    def show_logs(self):
        with dpg.window(label="Logger", tag="logger_window", width=400, height=600, pos=[0, 200]):
            dpg.add_text("", tag="log_output")

    def update(self, system_state: dict):
        self.scene.draw(system_state)
        
        # Always update the devices page
        self.devices_page.update(system_state)
        
        if dpg.does_item_exist("Configuration"):
            self.configuration_page.update(system_state)
        if dpg.does_item_exist("Calibration"):
            self.calibration_page.update(system_state)
        self.update_logs()

    def clear(self):
        pass

    def update_logs(self):
        log_text = self.gui_manager.get_latest_logs()
        dpg.set_value("log_output", log_text)

    def refresh(self):
        self.gui_manager.refresh_system()

class GuiManager:
    def __init__(self, pipe, logging_queue):
        self._pipe = pipe
        self._logging_queue = logging_queue
        self._server_config = None
        self._pages = {}
        self.log_messages = []
        self.max_log_messages = 1000  # Limit the number of stored messages

    def add_log(self, message, level=logging.INFO):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {logging.getLevelName(level)} - {message}"
        self.log_messages.insert(0, log_entry)  # Insert at the beginning
        
        # Limit the number of stored messages
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages.pop()  # Remove the oldest message

    def get_latest_logs(self):
        return "\n".join(self.log_messages[:100])  # Get the 100 most recent messages

    def clear_logs(self):
        self.log_messages.clear()

    def process_log_queue(self):
        while not self._logging_queue.empty():
            try:
                record = self._logging_queue.get_nowait()
                self.add_log(record.getMessage(), record.levelno)
            except queue.Empty:
                break

    def start(self):
        dpg.create_context()
        
        # Create all pages
        self._pages['visualization'] = VisualizationPage(self)
        self._pages['devices'] = DevicesPage('Devices', self)
        
        # Show all pages
        for page in self._pages.values():
            page.show()
        
        # Ensure the devices page is visible
        self._pages['devices'].show()
        
        # Create the viewport
        dpg.create_viewport(title="Vive Tracker Visualization", width=1920, height=1080)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()
        
        # Main loop
        while dpg.is_dearpygui_running():
            try:
                self.on_render()
                dpg.render_dearpygui_frame()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
        
        dpg.destroy_context()

    def on_render(self):
        self.process_log_queue()  # Process any new log messages
        
        system_state = {}
        while self._pipe.poll():
            data = self._pipe.recv()
            if "state" in data:
                system_state = data["state"]
            if "config" in data:
                self._server_config = data["config"]
        
        self._pages['visualization'].update(system_state)

    def get_config(self):
        return self._server_config

    def update_config(self, config):
        self._server_config = config

    def refresh_system(self):
        self._pipe.send({"refresh": None})








