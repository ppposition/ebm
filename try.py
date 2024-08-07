def binary_search(l, a):
    left, right = 0, len(l) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if l[mid] < a:
            left = mid + 1
        else:
            right = mid - 1
    return left  # 返回的是不小于a的第一个元素的索引

l = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 示例有序列表
a = 5  # 示例阈值

# 使用二分查找找到不小于a的第一个元素的索引
index = binary_search(l, a)

# 取出所有小于a的值
filtered_list = l[:index]

print(filtered_list)  # 输出: [1, 2, 3, 4]
