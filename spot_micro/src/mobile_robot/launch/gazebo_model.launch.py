import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # this name has to match the robot name in the Xacro file 
    robot_name_in_xacro='differential_drive_robot'
    
    # this is the name of our package, at the same time this is the name of the # folder that will be used to define the paths
    pakage_name = 'mobile_robot'

    #this is a relative path to the xacro file defining the model 
    relative_path_model_xacro = 'models/spotmicroai.xacro' 
    
    # uncom this if you want to define your own empty world model # however, then you have to create empty world.world
    #this is a relative path to the Gazebo world file # worldFileRelativePath = 'model/empty_world.world'
    
    # this is the absolute path to the model 
    path_model_xacro = os.path.join(get_package_share_directory(pakage_name), relative_path_model_xacro)
    # uncomment this if you are using your own world model
    # this is the absolute path to the world model

    #get the robot description from the xacro model file
    robot_description = xacro.process_file(path_model_xacro).toxml()

    #launch file from the gazebo_ros package
    gazebo_ros_pakage_launch = PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ros_gz_sim'),
                                                                        'launch', 'gz_sim.launch.py'))

    #launch description: this is if you are using an empty world model
    gazebo_launch = IncludeLaunchDescription(gazebo_ros_pakage_launch, launch_arguments = {'gz_args': ['-r -v -v4 empty.sdf'], 'on_exit_shutdown': 'true'}.items())
    
    # Gazebo node
    node_spawn_model_gazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robot_name_in_xacro,
            '-topic', 'robot_description',
            '-x', '1.0',  # Initial x position
            '-y', '2.0',  # Initial y position
            '-z', '0.5',  # Initial z position (height from the ground)
            '-R', '0.0',  # Initial roll (rotation around x-axis)
            '-P', '0.0',  # Initial pitch (rotation around y-axis)
            '-Y', '0.0'  # Initial yaw (rotation around z-axis, in radians)
        ],
        output='screen',
    )


    #Robot state publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,
        'use_sim_time':True}]
    )

    #this is very important so we can control the robot from ROS2
    bridge_params = os.path.join(
    get_package_share_directory(pakage_name),
    'parameters',
    'bridge_parameters.yaml'
    )


    start_gazebo_ros_bridge_cmd = Node(
        package = 'ros_gz_bridge',
        executable = 'parameter_bridge',
        arguments =[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output = 'screen',
    )

    # here we create an empty launch description object 
    launchDescriptionObject = LaunchDescription()

    # we add gazeboLaunch
    launchDescriptionObject.add_action(gazebo_launch)

    #we add the two nodes
    launchDescriptionObject.add_action(node_spawn_model_gazebo)
    launchDescriptionObject.add_action(node_robot_state_publisher) 
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)


    return launchDescriptionObject