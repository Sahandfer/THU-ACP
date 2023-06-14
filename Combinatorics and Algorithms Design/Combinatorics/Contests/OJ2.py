def find_num_ways(a, b, l, numWays = 0, visited =[]):
    if (l < 2):
        return 0
    skip = []
    for i in range(10):
        temp = []
        for j in range(10):
            temp.append(0)
        skip.append(temp)
    skip[1][3] = skip[3][1] = 2;
    skip[1][7] = skip[7][1] = 4;
    skip[3][9] = skip[9][3] = 6;
    skip[7][9] = skip[9][7] = 8;
    skip[1][9] = skip[9][1] = skip[2][8] = skip[8][2] = skip[3][7] = skip[7][3] = skip[4][6] = skip[6][4] = 5;

    neighbors = {
        1: [2, 4, 5, 6, 8],
        2: [1, 3, 4, 5, 6, 7, 9],
        3: [2, 4, 5, 6, 8],
        4: [1, 2, 3, 5, 7, 8, 9],
        5: [1, 2, 3, 4, 6, 7, 8, 9],
        6: [1, 2, 3, 5, 7, 8, 9],
        7: [2, 4, 5, 6, 8],
        8: [1, 3, 4, 5, 6, 7, 9],
        9: [2, 4, 5, 6, 8]
    }
    res = []
    paths = find_paths(neighbors, a, b, l, skip)

    return len(paths)
    
def find_paths(lockpad, start, end, l, skip, path=[]):
    path = path+[start]
    if (start == end):
        return [path]
    if (len(path) > l):
        return []
    paths=[]
    for lockKey in range(1,10):
        if (lockKey not in path):
            if (lockKey in lockpad[start]) or (skip[start][lockKey] in path):
                newpaths = find_paths(lockpad, lockKey, end, l, skip, path)
                for newpath in newpaths:
                    if(len(newpath) == l):
                        paths.append(newpath)
    return paths

if __name__ == "__main__":
    a, b, l = map(int, input().split)
    num_ways = find_num_ways(a,b,l)