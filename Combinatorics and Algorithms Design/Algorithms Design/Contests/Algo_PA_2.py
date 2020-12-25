import math


def print_neatly(n, m, words):
    """
    Table for costs, optimal cost would be at the last location (n+1th)
    There is an extra entry at the beginning for easier calculations in below steps
    """
    costs = [0 for i in range(n+1)]

    for i in range(1, n+1):
        """"
        The maximum cost was set to infinity to make the process of filtering
        out non-optimal costs in the following steps easier.
        """
        costs[i] = float('inf')
        for j in range(max(1, i-(m//2)+1), i+1):
            # 1. Calculate the number of spaces
            # Num of extra spaces at the end of the line
            extra_spaces = m-len(words[j-1])
            for k in range(j, i):
                # Add the spaces between words
                extra_spaces -= len(words[k]) + 1

            # 2. Calculate the cost of each line
            # Negative spaces means words won't fit on the line (unacceptable)
            if extra_spaces < 0:
                line_costs = float('inf')
            # We do not include the last line in the sum of cubes
            elif ((extra_spaces >= 0) and (i-1 == n-1)):
                line_costs = 0
            # Cube of the number of extra spaces to be included in the sum
            else:
                line_costs = int(math.pow(extra_spaces, 3))

            # 3. Calculate a potential optimal cost for lines until line i
            costs_temp = costs[j-1] + line_costs
            if (costs_temp < costs[i]):
                costs[i] = costs_temp  # the optimal cost

    print(costs[n])


if __name__ == "__main__":
    n, m = map(int, input().split())
    words = input().split()
    # n, m = 7, 10
    # words = "word like first as the the complete\n".split()
    print_neatly(n, m, words)
