# 定义动作名称
action_names = ["turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

# action_names = ["turn_over", "legs_stretch", "hands_stretch",
#           "legs_twitch", "hands_twitch", "head_move", "grasp", "kick", "still", "sit_lay"]


# 定义特征名称
# feature_names = ["StandardDeviationModule", "VarianceModule", "AverageModule"]
# feature_names = ["StandardDeviationModule", "AverageModule"]
feature_names = ["RangeModule", "EnergyModule", "StandardDeviationModule",
                 "RMSModule", "VarianceModule", "AverageModule"]