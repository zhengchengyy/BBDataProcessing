rule = "MR1:IF SD <= 0.215 AND Mean <= 0.538 AND SD <= 0.117 AND SD <= 0.049 AND Mean <= 0.423 THEN action_num = 2, action_proba = 0.9210526315789473"
index = rule.rfind(" ")
proba = rule[index+1:]
print(index)
print(proba)