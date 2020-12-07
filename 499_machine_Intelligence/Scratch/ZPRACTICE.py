nums = []
for x in range(10):
    if x < 2:
        nums.append(1)
    else:
        nums.append(nums[x-1])
        nums[x] += nums[x-2]
    print(str(x) + ":" + str(nums[x]))


