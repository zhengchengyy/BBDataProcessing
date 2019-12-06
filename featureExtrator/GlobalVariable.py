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

# action_names = [
#           "get_up","go_to_bed",
#           "turn_over","legs_stretch","hands_stretch",
#           "head_move","legs_move","hands_move",
#           "kick","legs_tremble","hands_tremble"]

action_names = [
          "get_up","go_to_bed",
          "turn_over","legs_stretch","hands_stretch",
          "head_move","legs_move","hands_move",
          "kick","legs_tremble","hands_tremble"]

# 定义特征名称
# feature_names = ["SDModule", "VarianceModule", "MeanModule"]

# feature_names = ["RangeModule", "EnergyModule", "SDModule",
#                  "RMSModule", "VarianceModule", "MeanModule", "MaxModule", "MinModule"]

# feature_names = ["RangeModule", "MeanModule", "SDModule", "RMSModule", "EnergyModule"]

feature_names = ["MeanModule", "SDModule"]

# feature_names = ["RangeModule", "EnergyModule", "SDModule", "FDEModule",
#                  "RMSModule", "VarianceModule", "MeanModule"]

# feature_names = ["RangeModule", "MeanModule", "SDModule", "EnergyModule", "FDEModule",
#                  "RMSModule"]

# feature_names = ["MeanModule", "SDModule", "EnergyModule","RMSModule"]