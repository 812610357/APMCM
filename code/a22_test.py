'''sto = list([0, 0, 0, 0])
for i in range(2, 4):
    sto[i] += 1
'''
# [[x1, y1][x2, y2][][]] x表示爹，y表示辈分
parent = list([])
for i in range(4):
    parent.append([1, 1])
print(parent)
