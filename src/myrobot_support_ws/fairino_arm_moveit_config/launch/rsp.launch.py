from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            "fairino_arm_moveit_descriptions", package_name="fairino_arm_moveit_config"
        )
        .trajectory_execution(file_path="config/moveit_controllers_real.yaml")
        .to_moveit_configs()
    )
    return generate_rsp_launch(moveit_config)
