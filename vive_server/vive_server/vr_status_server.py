from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Union
from triad_openvr import TriadOpenVR
from pydantic import BaseModel, Field
import asyncio
import json
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="VR Status API",
    description="""
    API for querying VR device status.
    
    WebSocket Endpoint:
    ------------------
    Connect to `ws://host:port/ws` to receive real-time device status updates.
    
    The WebSocket will send messages in one of these formats:
    1. Pose Update: { "poses": [ { "name": string, "pose_matrix": number[12] } ] }
    2. Error: { "error": string }
    
    The pose_matrix is a 3x4 transformation matrix flattened to:
    [m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23]
    """,
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Exposes all headers
)

vr = None

@app.on_event("startup")
async def startup_event():
    global vr
    try:
        vr = TriadOpenVR()
    except Exception as e:
        print(f"Failed to initialize OpenVR: {e}")

class DevicePose(BaseModel):
    name: str
    pose_matrix: List[float] = Field(
        default_factory=list,
        description="3x4 transformation matrix flattened to [m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23]"
    )
    is_recording: bool = Field(
        default=False,
        description="Whether this device is currently being recorded"
    )

# Define response models
class DeviceList(BaseModel):
    tracking_reference: List[str]
    hmd: List[str]
    controller: List[str]
    tracker: List[str]
    
class StatusResponse(BaseModel):
    devices: DeviceList
    poses: List[DevicePose]

class WebSocketError(BaseModel):
    error: str

class WebSocketPoseUpdate(BaseModel):
    poses: List[DevicePose]

class RecordingRequest(BaseModel):
    device_ids: List[str] = Field(..., description="The IDs of the devices to record")
    fixed_device_ids: List[str] = Field(..., description="The IDs of the devices that are fixed")

class StopRecordingRequest(BaseModel):
    device_ids: List[str] = Field(..., description="The IDs of the devices to stop recording")
    download: bool = Field(default=False, description="If true, returns the recorded data as JSON")

# This union type represents all possible message types that can be sent over the websocket
WebSocketMessage = Union[WebSocketError, WebSocketPoseUpdate]

# Replace the single recording state with a dictionary of recording states per device
recording_states = {}  # device_id -> recording state dictionary

@app.post("/start-recording")
async def start_recording(request: RecordingRequest):
    if not vr:
        raise HTTPException(status_code=503, detail="OpenVR not initialized")
    
    # Check if any of the requested devices are already recording
    already_recording = [device_id for device_id in request.device_ids if device_id in recording_states]
    if already_recording:
        raise HTTPException(status_code=400, detail=f"Already recording devices: {', '.join(already_recording)}")
    
    # Verify all devices exist
    missing_devices = [device_id for device_id in request.device_ids if device_id not in vr.devices]
    if missing_devices:
        raise HTTPException(status_code=404, detail=f"Devices not found: {', '.join(missing_devices)}")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("~/Documents/tracker_data").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"multi_device_recording_{timestamp}.txt"
        output_path = output_dir / filename

        shared_file = output_path.open('w')
        
        # Store the same file handle for each device
        for device_id in request.device_ids:
            recording_states[device_id] = {
                "output_file": shared_file,
                "output_path": output_path,
                "is_fixed": device_id in request.fixed_device_ids
            }
        
        return {"message": f"Started recording devices: {', '.join(request.device_ids)} to {filename}"}
    
    except Exception as e:
        # Clean up if error occurs
        for device_id in request.device_ids:
            if device_id in recording_states:
                try:
                    recording_states[device_id]["output_file"].close()
                    del recording_states[device_id]
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {str(e)}")

@app.post("/stop-recording")
async def stop_recording(request: StopRecordingRequest):
    not_recording = [device_id for device_id in request.device_ids if device_id not in recording_states]
    if not_recording:
        raise HTTPException(status_code=400, detail=f"Devices not being recorded: {', '.join(not_recording)}")
    
    try:
        # Get the first device's file info (they all share the same file)
        first_device = request.device_ids[0]
        shared_file = recording_states[first_device]["output_file"]
        filepath = recording_states[first_device]["output_path"]
        
        # Close the shared file
        shared_file.close()
        
        # Read the file contents
        with open(filepath, 'r') as f:
            content = f.read()
            # Remove trailing comma and newline, then wrap in brackets
            content = '[' + content.rstrip(',\n') + ']'
            
        # Clean up recording states for all devices
        for device_id in request.device_ids:
            del recording_states[device_id]
        
        # Return either the JSON data or a success message
        if request.download:
            return json.loads(content)  # Returns the parsed JSON array
        else:
            return {"message": f"Recording stopped and saved to {filepath}"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_poses: Dict[str, List[float]] = {}
    
    try:
        while True:
            if not vr:
                error_msg = WebSocketError(error="OpenVR not initialized")
                await websocket.send_json(error_msg.dict())
                await asyncio.sleep(1)
                continue
            
            try:
                changed_poses = []

                for device_name, device in vr.devices.items():
                    pose_matrix = device.get_pose_matrix()
                    if pose_matrix is not None:
                        # Flatten matrix to list
                        matrix_list = [
                            pose_matrix[0][0], pose_matrix[0][1], pose_matrix[0][2], pose_matrix[0][3],
                            pose_matrix[1][0], pose_matrix[1][1], pose_matrix[1][2], pose_matrix[1][3],
                            pose_matrix[2][0], pose_matrix[2][1], pose_matrix[2][2], pose_matrix[2][3]
                        ]
                        
                        # Compare with last pose
                        if (device_name not in last_poses or 
                            any(abs(a - b) > 1e-5 for a, b in zip(matrix_list, last_poses[device_name]))):
                            
                            last_poses[device_name] = matrix_list
                            changed_poses.append(DevicePose(name=device_name, pose_matrix=matrix_list))
                            
                            # If we're recording this device, write to file
                            if device_name in recording_states:
                                current_timestamp = str(datetime.now())
                                message = json.dumps(
                                    {
                                        "device_id": device_name,
                                        "is_fixed": recording_states[device_name]["is_fixed"],
                                        "ts": current_timestamp, 
                                        "pose": matrix_list
                                    },
                                    separators=(',', ':')
                                )
                                recording_states[device_name]["output_file"].write(message + ",\n")
                                recording_states[device_name]["output_file"].flush()  # Ensure data is written immediately
                
                if changed_poses:
                    update = WebSocketPoseUpdate(poses=changed_poses)
                    await websocket.send_json(update.dict())
                
            except Exception as e:
                error_msg = WebSocketError(error=f"Error getting device status: {str(e)}")
                await websocket.send_json(error_msg.dict())
            
            await asyncio.sleep(1/120)  # Update rate of 120Hz
            
    except Exception as e:
        print(f"WebSocket connection closed: {e}")

# Keep the REST endpoint as well
@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    if not vr:
        raise HTTPException(status_code=503, detail="OpenVR not initialized")
    
    try:
        device_poses = []
        for device_name, device in vr.devices.items():
            pose_matrix = device.get_pose_matrix()
            if pose_matrix is not None:
                # Flatten the 3x4 matrix into a list of 12 elements
                matrix_list = [
                    pose_matrix[0][0], pose_matrix[0][1], pose_matrix[0][2], pose_matrix[0][3],
                    pose_matrix[1][0], pose_matrix[1][1], pose_matrix[1][2], pose_matrix[1][3],
                    pose_matrix[2][0], pose_matrix[2][1], pose_matrix[2][2], pose_matrix[2][3]
                ]
                device_poses.append(DevicePose(
                    name=device_name, 
                    pose_matrix=matrix_list,
                    is_recording=device_name in recording_states
                ))

        return StatusResponse(
            devices=DeviceList(
                tracking_reference=vr.object_names["Tracking Reference"],
                hmd=vr.object_names["HMD"],
                controller=vr.object_names["Controller"],
                tracker=vr.object_names["Tracker"]
            ),
            poses=device_poses
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting device status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)