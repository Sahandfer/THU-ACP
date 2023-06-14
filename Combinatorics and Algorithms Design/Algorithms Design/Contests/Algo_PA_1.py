def merge(start, mid, end, p_sums):
    l_arr = p_sums[start:mid]  # Left sub-array
    r_arr = p_sums[mid:end]  # Right sub-array

    i = j = 0 # initial indexes of the left and right aub-array respectively
    idx = start

    while i < len(l_arr) and j < len(r_arr):
        if l_arr[i] < r_arr[j]:
            p_sums[idx] = l_arr[i]
            i += 1
        else:
            p_sums[idx] = r_arr[j]
            j += 1
        idx += 1

    # Copying remaining elements
    while i < len(l_arr):
        p_sums[idx] = l_arr[i]
        i += 1
        idx += 1
    while j < len(r_arr):
        p_sums[idx] = r_arr[j]
        j += 1
        idx += 1


def merge_sort(start, end, p_sums):
    mid = start + (end - start)//2
    if (start == mid) or (start == end) or (mid == end):
        return 0

    # MergeSort stage
    num_sums = merge_sort(start, mid, p_sums) + merge_sort(mid, end, p_sums)

    i = j = mid

    for k in range(start, mid):
        # The range sum is lower than the lower bound (not acceptable)
        while i < end and p_sums[i] - p_sums[k] < lower:
            i += 1
        # The range sum is lower than the upper bound (acceptable)
        while j < end and p_sums[j] - p_sums[k] <= upper:
            j += 1
        # Therefore, number of acceptable range sums is
        num_sums += j-i

    # Merge stage
    merge(start, mid, end, p_sums)

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
    # n, lower, upper, nums = 3, -2, 2, [-2, 5, -1]
    num_sum_ranges()
