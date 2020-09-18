def calc_bi_coeffs(a, b):
    output = 1
    if (b < 0 or b > a):
        output = 0
    elif (b==0 or a == b):
        output = 1
    else:
        for c in range(min(b, a-b)):
            output *= a
            output //= c+1
            a-=1
    return output
            

if __name__ == "__main__":
    a, b = map(int, input().split())
    print(calc_bi_coeffs(a,b))