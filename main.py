# This is a sample Python script.
import sys
from collections import deque
from typing import List

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n = len(stoneValue)

        dp = [-sys.maxsize for _ in range(n + 1)]
        dp[n] = 0
        dp[n - 1] = stoneValue[n - 1]

        for i in range(n - 2, -1, -1):
            temp = 0
            for j in range(min(3, n- i)):
                temp += stoneValue[i + j]
                dp[i] = max(dp[i], temp - dp[i + 1 + j])


        print(dp)

        return "Tie" if dp[0] == 0 else ("Bob" if dp[0] < 0 else "Alice")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solution = Solution()
    print(solution.stoneGameIII([1,2,3,7]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
