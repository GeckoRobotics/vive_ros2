#!/usr/bin/env python3
# Including ros related libraries.
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
import tf2_ros
import geometry_msgs.msg
import pdb

# Including threading and comms related system libraries.
from threading import Thread, Event
from queue import Queue
import socket

# Including HTC vive related libraries.
from vive_tracker_client import ViveTrackerClient


class ViveTrackerNode(Node):

    def __init__(self):
        super().__init__('vive_tracker_node')
        # self.declare_parameter('host_ip', '192.168.50.171')
        self.declare_parameter('host_ip', socket.gethostbyname(socket.gethostname()))
        self.declare_parameter('host_port', 8000)
        self.declare_parameter('tracker_name', 'tracker_1')
        self.declare_parameter('topic', 'tracker/odom')
        self.declare_parameter('link_name', 'odom')
        self.declare_parameter('child_link_name', 'tracker_link')

        (self.host_ip, self.host_port, self.tracker_name, self.link_name, self.child_link_name, self.topic) = self.get_parameters(
            ['host_ip', 'host_port', 'tracker_name', 'link_name', 'child_link_name', 'topic'])

        topic = self.topic.get_parameter_value().string_value
        topic_name = self.tracker_name.get_parameter_value().string_value + '/odom' if topic == "" else topic
        self.odom_pub = self.create_publisher(Odometry, topic_name,
            qos_profile=qos_profile_sensor_data)
        self.bs_odom_pub = self.create_publisher(Odometry, topic_name,
            qos_profile=qos_profile_sensor_data)

        self.client = ViveTrackerClient(host=self.host_ip.get_parameter_value().string_value,
                                   port=self.host_port.get_parameter_value().integer_value,
                                   tracker_name=self.tracker_name.get_parameter_value().string_value,
                                   should_record=False)

        self.message_queue = Queue()
        self.message_queue_ID = Queue()
        self.message_queue_bs = Queue()
        self.message_queue_bs_IDs = Queue()
        self.kill_thread = Event()
        self.client_thread = Thread(target=self.client.run_threaded, args=(self.message_queue, self.message_queue_ID, self.message_queue_bs, self.message_queue_bs_IDs, self.kill_thread,))
        self.br = tf2_ros.TransformBroadcaster(self)

    def read_odom(self):
        try:
            self.client_thread.start()
            #self.client.update()
            while rclpy.ok():
                msg = self.message_queue.get()
                msg_ID = self.message_queue_ID.get()
                odom_msg = self.set_odom(msg,msg_ID)            
                self.odom_pub.publish(odom_msg)
                self.publish_tf(odom_msg.pose.pose, odom_msg.header.frame_id, odom_msg.child_frame_id)

                msg_bs_ID = self.message_queue_bs_IDs.get()
                msg_bs = self.message_queue_bs.get()
                bs_odom_msg = self.set_bs_odom(msg_bs, msg_bs_ID)
                for i in range(0,len(bs_odom_msg)):
                    self.bs_odom_pub.publish(bs_odom_msg[i])
                    self.publish_tf(bs_odom_msg[i].pose.pose, bs_odom_msg[i].header.frame_id, bs_odom_msg[i].child_frame_id)

                wf_odom_msg = self.set_world_convention_odom()
                self.publish_tf(wf_odom_msg.pose.pose, wf_odom_msg.header.frame_id, wf_odom_msg.child_frame_id)



        finally:
            # cleanup
            self.kill_thread.set()
            self.client_thread.join()
    
    def set_odom(self, msg, msg_ID):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'world_vive'

        odom_msg.child_frame_id = msg_ID    #self.child_link_name.get_parameter_value().string_value

        odom_msg.pose.pose.position.x = msg.x
        odom_msg.pose.pose.position.y = msg.y
        odom_msg.pose.pose.position.z = msg.z

        odom_msg.pose.pose.orientation.x = msg.qx
        odom_msg.pose.pose.orientation.y = msg.qy
        odom_msg.pose.pose.orientation.z = msg.qz
        odom_msg.pose.pose.orientation.w = msg.qw

        odom_msg.twist.twist.linear.x = msg.vel_x
        odom_msg.twist.twist.linear.y = msg.vel_y
        odom_msg.twist.twist.linear.z = msg.vel_z

        odom_msg.twist.twist.angular.x = msg.p
        odom_msg.twist.twist.angular.y = msg.q
        odom_msg.twist.twist.angular.z = msg.r

        return odom_msg
    
    def set_world_convention_odom(self):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'world_vive_ros'

        odom_msg.child_frame_id = 'world_vive'    #self.child_link_name.get_parameter_value().string_value

        odom_msg.pose.pose.position.x = 0.0
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation.x = 0.707108
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = 0.0
        odom_msg.pose.pose.orientation.w = 0.707105

        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0

        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        return odom_msg
    
    def set_bs_odom(self, msg_bs, msg_bs_ID):
        bs_odom_msg = []
        for i in range(0,len(msg_bs_ID)):
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            # if i==0:
            #     odom_msg.header.frame_id = "world_vive"
            # else:
            #     odom_msg.header.frame_id = msg_bs_ID[0]
            odom_msg.header.frame_id = "world_vive"

            odom_msg.child_frame_id = msg_bs_ID[i]
            msg = msg_bs[i]

            odom_msg.pose.pose.position.x = msg.x
            odom_msg.pose.pose.position.y = msg.y
            odom_msg.pose.pose.position.z = msg.z

            odom_msg.pose.pose.orientation.x = msg.qx
            odom_msg.pose.pose.orientation.y = msg.qy
            odom_msg.pose.pose.orientation.z = msg.qz
            odom_msg.pose.pose.orientation.w = msg.qw
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0

            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0
            bs_odom_msg.append(odom_msg)
        #print(len(bs_odom_msg), "Base station odom", bs_odom_msg)
        return bs_odom_msg
        

    def publish_tf(self, odom_pose, parent_frame, child_frame):
        tf_odom = geometry_msgs.msg.TransformStamped()
        tf_odom.header.stamp = self.get_clock().now().to_msg()
        tf_odom.header.frame_id =  parent_frame
        tf_odom.child_frame_id = child_frame
        tf_odom.transform.translation.x = odom_pose.position.x
        tf_odom.transform.translation.y = odom_pose.position.y
        tf_odom.transform.translation.z = odom_pose.position.z
        tf_odom.transform.rotation.x = odom_pose.orientation.x
        tf_odom.transform.rotation.y = odom_pose.orientation.y
        tf_odom.transform.rotation.z = odom_pose.orientation.z
        tf_odom.transform.rotation.w = odom_pose.orientation.w
        self.br.sendTransform(tf_odom)

def main(args=None):
    rclpy.init(args=args)
    vive_tracker_node = ViveTrackerNode()
    #pdb.set_trace()

    rclpy.spin(vive_tracker_node.read_odom())
    vive_tracker_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
