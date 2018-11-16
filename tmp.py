"""
Your rank will rise at least {num_team_onshot_zero}, if you are not lazy.

From public leader board, we can get the number of the teams who submit only one submission, whose score is zero.
The number is {num_team_onshot_zero} at the time I post this.
So, I think Your rank will rise at least {num_team_onshot_zero} because {num_team_onshot_zero} team will shake down due to their wrong submission, if you are not in those teams.
Is this prediction wrong?

BTW, I used the code below.
~~~```
import pandas as pd
data = pd.read_csv("ga-customer-revenue-prediction-publicleaderboard.csv")
data_agg = data.groupby("TeamName").agg({"Score":["min", "count"]})
num_team_onshot_zero = data_agg[(data_agg["Score"]["min"] == 0) & (data_agg["Score"]["count"]==1)].shape[0]
print("The number of the teams who submit only one submission, whose score is zero:", num_team_onshot_zero)
```~~~

LB の Raw Data ダウンロードは、最高記録のみ書き込まれている。
つまり、はじめに 0.0000 を叩き出したユーザーは、その後いくら Submit しても、1行しか抽出されない。・・・人力で数えるしかない。。
"""
