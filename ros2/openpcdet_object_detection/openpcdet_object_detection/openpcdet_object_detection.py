from typing import Any, Optional, Union

from perception_msgs.msg import HEXAMOTION, Object, ObjectList
import perception_msgs_utils as pmu
from sensor_msgs.msg import PointCloud2
import rclpy
from rclpy.node import Node
import rclpy.exceptions
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult


class OpenpcdetObjectDetection(Node):

    def __init__(self):
        """Constructor"""
        super().__init__("openpcdet_object_detection")

        self.subscriber = None

        self.publisher = None

        self.auto_reconfigurable_params: list[str] = []
        self.param = self.declare_and_load_parameter(
            name="param",
            param_type=rclpy.Parameter.Type.DOUBLE,
            description="TODO",
            default=1.0,
            from_value=0.0,
            to_value=10.0,
            step_value=0.1,
        )

        self.setup()

    def declare_and_load_parameter(
        self,
        name: str,
        param_type: rclpy.Parameter.Type,
        description: str,
        default: Optional[Any] = None,
        add_to_auto_reconfigurable_params: bool = True,
        is_required: bool = False,
        read_only: bool = False,
        from_value: Optional[Union[int, float]] = None,
        to_value: Optional[Union[int, float]] = None,
        step_value: Optional[Union[int, float]] = None,
        additional_constraints: str = "",
    ) -> Any:
        """Declares and loads a ROS parameter

        Args:
            name (str): name
            param_type (rclpy.Parameter.Type): parameter type
            description (str): description
            default (Optional[Any], optional): default value
            add_to_auto_reconfigurable_params (bool, optional): enable reconfiguration of parameter
            is_required (bool, optional): whether failure to load parameter will stop node
            read_only (bool, optional): set parameter to read-only
            from_value (Optional[Union[int, float]], optional): parameter range minimum
            to_value (Optional[Union[int, float]], optional): parameter range maximum
            step_value (Optional[Union[int, float]], optional): parameter range step
            additional_constraints (str, optional): additional constraints description

        Returns:
            Any: parameter value
        """

        # declare parameter
        param_desc = ParameterDescriptor()
        param_desc.description = description
        param_desc.additional_constraints = additional_constraints
        param_desc.read_only = read_only
        if from_value is not None and to_value is not None:
            if param_type == rclpy.Parameter.Type.INTEGER:
                range = IntegerRange(from_value=from_value, to_value=to_value)
                if step_value is not None:
                    range.step = step_value
                param_desc.integer_range = [range]
            elif param_type == rclpy.Parameter.Type.DOUBLE:
                range = FloatingPointRange(from_value=from_value, to_value=to_value)
                if step_value is not None:
                    range.step = step_value
                param_desc.floating_point_range = [range]
            else:
                self.get_logger().warn(f"Parameter type of parameter '{name}' does not support specifying a range")
        self.declare_parameter(name, param_type, param_desc)

        # load parameter
        try:
            param = self.get_parameter(name).value
            self.get_logger().info(f"Loaded parameter '{name}': {param}")
        except rclpy.exceptions.ParameterUninitializedException:
            if is_required:
                self.get_logger().fatal(f"Missing required parameter '{name}', exiting")
                raise SystemExit(1)
            else:
                self.get_logger().warn(f"Missing parameter '{name}', using default value: {default}")
                param = default
                self.set_parameters([rclpy.Parameter(name=name, value=param)])

        # add parameter to auto-reconfigurable parameters
        if add_to_auto_reconfigurable_params:
            self.auto_reconfigurable_params.append(name)

        return param

    def parameters_callback(self, parameters: list[rclpy.Parameter]) -> SetParametersResult:
        """Handles reconfiguration when a parameter value is changed

        Args:
            parameters (list[rclpy.Parameter]): parameters

        Returns:
            SetParametersResult: parameter change result
        """

        for param in parameters:
            if param.name in self.auto_reconfigurable_params:
                setattr(self, param.name, param.value)
                self.get_logger().info(f"Reconfigured parameter '{param.name}' to: {param.value}")

        result = SetParametersResult()
        result.successful = True

        return result

    def setup(self):
        """Sets up subscribers, publishers, etc. to configure the node"""

        # callback for dynamic parameter configuration
        self.add_on_set_parameters_callback(self.parameters_callback)

        # subscriber for handling incoming messages
        self.subscriber = self.create_subscription(PointCloud2, "~/input", self.topic_callback, qos_profile=10)
        self.get_logger().info(f"Subscribed to '{self.subscriber.topic_name}'")

        # publisher for publishing outgoing messages
        self.publisher = self.create_publisher(ObjectList, "~/output", qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher.topic_name}'")

    def topic_callback(self, point_cloud: PointCloud2):
        """Processes messages received by a subscriber

        Args:
            point_cloud (PointCloud2): message
        """

        self.get_logger().info(f"Message received with stamp: '{point_cloud.header.stamp}'")

        # TODO implement object detection using OpenPCDet
        # objects = detect(point_cloud)

        object_list_msg = ObjectList()
        object_list_msg.header = point_cloud.header

        # for obj in objects:

        object_msg = Object()
        pmu.initialize_state(object_msg, HEXAMOTION.MODEL_ID)
        object_msg.state.continuous_state[HEXAMOTION.X] = 1.0
        object_msg.state.continuous_state[HEXAMOTION.Y] = 1.0
        object_msg.state.continuous_state[HEXAMOTION.Z] = 1.0
        object_msg.state.continuous_state[HEXAMOTION.ROLL] = 0.0
        object_msg.state.continuous_state[HEXAMOTION.PITCH] = 0.0
        object_msg.state.continuous_state[HEXAMOTION.YAW] = 1.0
        object_msg.state.continuous_state[HEXAMOTION.LENGTH] = 1.0
        object_msg.state.continuous_state[HEXAMOTION.WIDTH] = 1.0
        object_msg.state.continuous_state[HEXAMOTION.HEIGHT] = 1.0
        object_list_msg.objects.append(object_msg)

        self.publisher.publish(object_list_msg)


def main():

    rclpy.init()
    node = OpenpcdetObjectDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
