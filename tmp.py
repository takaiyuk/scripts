import pandas
data = pd.read_csv("ga-customer-revenue-prediction-publicleaderboard.csv")
data_agg = data.groupby("TeamName").agg({"Score":["min", "count"]})
num_team_onshot_zero) = data_agg[(data_agg["Score"]["min"] == 0) & (data_agg["Score"]["count"]==1)].shape[0]
print("The number of the teams who submit only one submission, whose score is zero:", num_team_onshot_zero)
