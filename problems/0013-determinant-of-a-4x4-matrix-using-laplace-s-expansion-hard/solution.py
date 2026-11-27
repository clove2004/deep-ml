def determinant_3x3(a: list[list[int|float]]) -> float:
    # Your recursive implementation here
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0])
def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # Your recursive implementation here
    ans = 0
    for j in range (4):
        cur = []
        for i in range (1, 4):
            c = []
            for k in range (0, 4):
                if k != j:
                    c.append(matrix[i][k])
            cur.append(c)
        ans += ((-1)**j)*matrix[0][j]*determinant_3x3(cur)
    return ans