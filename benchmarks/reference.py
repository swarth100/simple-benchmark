import math


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


def histogram(nums: list[int]):
    max_height = max(nums)

    # Printing the histogram from top to bottom
    for level in range(max_height, 0, -1):
        line = "|"
        for height in nums:
            # Add a '#' if the current column height is >= current level, else add '.'
            line += "#" if height >= level else "."
        print(line)

    # Printing the horizontal axis
    print("+" + "-" * len(nums))


def scatter(size: int, nums: list[tuple[int, int]]):
    # Create a set of nums for faster lookup
    points_set = set((x, y) for x, y in nums)

    # Iterate through each row (y-axis)
    for y in range(size, 0, -1):
        line = "|"
        for x in range(1, size + 1):
            # Check if the point is in the points list
            if (x, y) in points_set:
                line += "*"
            else:
                line += "."
        print(line)

    # Printing the horizontal axis
    print("+" + "-" * (size))


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


def generate_trajectory(velocity: float, angle: float, size: list[int]):
    g = 9.81  # Acceleration due to gravity (m/s^2)
    angle_rad = math.radians(angle)  # Convert angle to radians

    # Function to calculate the position at a given x
    def position(x):
        # Prevent division by zero issues!
        if velocity == 0 or (math.cos(angle_rad) == 0):
            return x, 0

        t = x / (velocity * math.cos(angle_rad))
        y = velocity * t * math.sin(angle_rad) - 0.5 * g * t**2
        # We round to ensure consistent mathematical charting
        return x, round(y)

    # Generate points for the trajectory
    points = [position(i) for i in range(size[0])]

    # Create the ASCII chart
    chart = [[" " for _ in range(size[0])] for _ in range(size[1])]
    for x, y in points:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            chart[y][x] = "#"

    # Add axes
    for i in range(size[0]):
        chart[0][i] = "-"
    for i in range(size[1]):
        chart[i][0] = "|"
    chart[-1][0] = "^"
    chart[0][0] = "+"
    chart[0][-1] = ">"

    # Print the chart in reverse order (to allow for bot-to-top printing)
    for row in reversed(chart):
        print("".join(row))
