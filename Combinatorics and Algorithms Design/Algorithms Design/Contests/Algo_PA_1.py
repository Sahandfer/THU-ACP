n, lower, upper = 0, 0 , 0
nums = []
p_arr = []


def merge_sort(start, end):
    mid = (start+end)/2
    if (start == end): return 0
    if (start == mid): return 0
    
    num_sum = merge_sort(start, mid) + merge_sort(mid, end)
    
    i = mid
    j = mid
    for val in p_arr[start:end]:
        while i < end and p_arr[i] < lower + val: i += 1
        while j < end and p_arr[j] < upper + val: j +=1
        num_sum += (j-i)
        
    p_arr[start:end] = sorted(p_arr[start:end])
    return num_sum

def num_range_sums(n, lower, upper, nums):
    p_arr = range(n)
    for i in range(1, n):
        p_arr[i] = p_arr[i-1]+nums[i]
    print(merge_sort(lower, upper, p_arr))
    

if __name__ == "__main__":
    # n, lower, upper = map(int, input().split())
    # nums = map(int, input().split())

    # num_range_sums(n, lower, upper, nums)
    num_range_sums(3,-2,-2,[-2, 5, -1])