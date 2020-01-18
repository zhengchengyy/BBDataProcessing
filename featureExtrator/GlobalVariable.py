# 定义动作名称
action_names = ["turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

# action_names = ["turn_over", "legs_stretch", "hands_stretch",
#           "legs_twitch", "hands_twitch", "head_move", "grasp", "kick", "still", "sit_lay"]

# action_names = [
#           "turn_over","legs_stretch","hands_stretch",
#           "legs_tremble","hands_tremble","body_tremble",
#           "head_move","legs_move","hands_move",
#           "hands_rising","kick"]

action_names = [
          "turn_over","legs_stretch","hands_stretch",
          "head_move","legs_move","hands_move",
          "kick","legs_tremble","hands_tremble"]

# action_names = [
#           "turn_over","legs_stretch","hands_stretch",
#           "head_move","legs_move","hands_move",
#           "kick","legs_tremble","hands_tremble",
#           "go_to_bed","get_up"]

# 定义特征名称
# feature_names = ["SDModule", "VarianceModule", "MeanModule"]

# feature_names = ["RangeModule", "EnergyModule", "SDModule",
#                  "RMSModule", "VarianceModule", "MeanModule", "MaxModule", "MinModule"]

# feature_names = ["RangeModule", "MeanModule", "SDModule", "RMSModule", "EnergyModule"]

# feature_names = ["RangeModule", "EnergyModule", "SDModule", "FDEModule",
#                  "RMSModule", "VarianceModule", "MeanModule"]

feature_names = ["MeanModule", "SDModule"]

# 不同设备特征重要性分布不一样
# feature_names = ["RangeModule", "MeanModule", "SDModule", "EnergyModule", "FDEModule",
#                  "RMSModule"]

# feature_names = ["MeanModule", "SDModule", "EnergyModule","RMSModule"]