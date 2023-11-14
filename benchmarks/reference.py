def square(size: int):
    for x in range(size):
        if (x == 0) or (x == size - 1):
            print("*" * size)
        else:
            print("*" + ("." * (size - 2)) + "*")


def triangle(size: int):
    for x in range(1, size + 1):
        n_stars = x
        n_dots = size - n_stars
        stars = n_stars * "*"
        dots = n_dots * "."
        print(dots + stars)


def rhombus(size: int):
    limit = (size + 1) // 2
    for start, end, step in [(0, limit, 1), (limit - 1 - (size % 2), -1, -1)]:
        for x in range(start, end, step):
            stars = 1 + (x * 2)
            dots = (size - stars) // 2
            row = "." * dots + "*" * stars + "." * dots
            print(row)


def running_sum(nums: list[int]) -> list[int]:
    result: list[int] = []
    total: int = 0
    for num in nums:
        total += num
        result.append(total)
    return result


def count_pairs(nums: list[int], k: int) -> int:
    count: int = 0
    n: int = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(nums[i] - nums[j]) == k:
                count += 1
    return count


def dividing_numbers(left: int, right: int) -> list[int]:
    def _is_self_dividing(num):
        original_num = num
        while num > 0:
            digit = num % 10
            if digit == 0 or original_num % digit != 0:
                return False
            num //= 10

        return True

    result = []

    for x in range(left, right + 1):
        if _is_self_dividing(x):
            result.append(x)

    return result
