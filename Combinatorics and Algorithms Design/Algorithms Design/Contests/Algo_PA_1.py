def merge_sort(start, end, p_sums):
    mid = start + (end- start)//2
    if (start == mid) or (start == end) or (mid == end): return 0

    # Divide the problem into 2 smaller subproblems
    num_sums = merge_sort(start, mid, p_sums) + merge_sort(mid, end, p_sums)
    
    # The merge stage
    i = j = mid
    # For each k in the left half, find number of i and j in the right half such that
    for k in range(start, mid):
        # The range sum is lower than the lower bound (not acceptable)
        while i < end and p_sums[i] - p_sums[k] < lower: i+=1
        # The range sum is lower than the upper bound (acceptable)
        while j < end and p_sums[j] - p_sums[k] <= upper: j+=1
        # Therefore, number of acceptable range sums is
        num_sums += j-i
    
    # The sort stage
    p_sums[start:end] = sorted(p_sums[start:end]) 
    
    return num_sums

def num_sum_ranges():
    # Initially we calculate the range sum for all available indices with lower bound 0 => S(0, i) = P[i] - P[-1]
    # Where P[i] = sum of all the numbers between 0 and indice i in nums
    p_sums = [0]
    for i in range(n):
        p_sums.append(p_sums[-1]+nums[i])

    # p_sums would have an extra element 0 in the its beginning compared to nums => len(p_sums) = n+1
    print(merge_sort(0, n+1, p_sums))

if __name__ == "__main__":
    global n, nums, lower, upper
    n, lower, upper = map(int, input().split())
    nums = list(map(int, input().split()))
    num_sum_ranges()
