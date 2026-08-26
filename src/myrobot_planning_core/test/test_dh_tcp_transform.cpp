#include <gtest/gtest.h>

#include "myrobot_planning_core/dh_kinematics.h"

namespace fairino_planning {

TEST(DhTcpTransform, PreservesEachUrdfToolOffset) {
    const DHParams params{};
    for (const double wrist3_to_tool_z : {0.11, 0.2168}) {
        Transform4d wrist3_to_tool = Transform4d::Identity();
        wrist3_to_tool(2, 3) = wrist3_to_tool_z;

        const Transform4d flange_to_tool =
            DHKinematics::flangeToToolTransform(params, wrist3_to_tool);
        EXPECT_NEAR(flange_to_tool(2, 3), wrist3_to_tool_z - params.d[5], 1e-12);

        const DHKinematics fk(params, flange_to_tool);
        EXPECT_NEAR(
            params.d[5] + fk.toolTransform(ToolModel::GRIPPER)(2, 3),
            wrist3_to_tool_z, 1e-12);
    }
}

}  // namespace fairino_planning
