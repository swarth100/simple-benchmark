import math

from pydantic import BaseModel


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


def scatter(size: int, points: list[tuple[int, int]]):
    # Create a set of points for faster lookup.
    # This step is optional, a list works as well.
    points_set = set((x, y) for x, y in points)

    # Iterate through each row (y-axis).
    # NOTE: Ensure we also print the direction of the axes.
    for y in range(size, 0, -1):
        line = "|" if y != size else "^"
        for x in range(1, size + 1):
            # Check if the point is in the points list
            if (x, y) in points_set:
                line += "*"
            else:
                line += "."
        print(line)

    # Printing the horizontal axis at the bottom
    print("+" + "-" * (size - 1) + ">")


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


def generate_trajectory(velocity: int, angle: int, size: list[int]):
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


def sort_by_height(names: list[str], heights: list[int]) -> list[str]:
    # Initialize a dictionary to map heights to names
    height_to_names = {}

    # Populate the dictionary.
    # The names are a list as multiple people can have the same height!
    for name, height in zip(names, heights):
        if height not in height_to_names:
            height_to_names[height] = []
        height_to_names[height].append(name)

    # Sort the dictionary by height in descending order and concatenate names
    sorted_names = []
    for height in sorted(height_to_names.keys(), reverse=True):
        sorted_names.extend(height_to_names[height])

    return sorted_names


def best_student(gradebook: dict[str, list[float]]) -> str:
    highest_avg = 0
    top_student = ""

    for student, grades in gradebook.items():
        # Calculate the average grade for the student
        avg_grade = sum(grades) / len(grades)

        # Check if this average is the highest so far
        if avg_grade > highest_avg:
            highest_avg = avg_grade
            top_student = student

    return top_student


def jobs_counter(jobs: list[str]) -> dict[str, int]:
    inventory = {}

    for item in jobs:
        if item in inventory:
            inventory[item] += 1
        else:
            inventory[item] = 1

    return inventory


def can_visit_all_rooms(unlocked: str, rooms: dict[str, list[str]]) -> bool:
    visited = []  # Keep track of visited rooms

    # Stack to keep track of rooms to visit, starting with room "0"
    stack = [unlocked]

    while stack:
        room = stack.pop()  # Get the current room
        if room in rooms:
            if room not in visited:
                visited.append(room)

                for key in rooms[room]:  # Explore keys in the current room
                    if key not in visited:  # If the room hasn't been visited yet
                        stack.append(key)  # Add to stack to visit later

    # Check if all rooms have been visited
    return len(visited) == len(rooms)


def complementary_dna(sequence: list[str]) -> list[str]:
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return [complement[nucleotide] for nucleotide in sequence]


def prime_factors(number: int) -> list[int]:
    factors = []
    divisor = 2
    while number > 1:
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        divisor += 1
    return factors


def partitioning_line(
    start: list[int], end: list[int], points: list[list[int]]
) -> list[list[int]]:
    def calculate_position(point, start, end):
        # The equation for the line through (x1, y1) and (x2, y2) is:
        # (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1) = 0
        # If the result is positive or zero, the point is on or above the line.
        x, y = point
        x1, y1 = start
        x2, y2 = end

        return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

    # Filter the points to include only those on or above the line
    return [point for point in points if calculate_position(point, start, end) >= 0]


def tartaglia_triangle(n: int) -> list[list[int]]:
    triangle = [[1]]
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        row.append(1)
        triangle.append(row)
    return triangle


def missing_triangle(sides: dict):
    c1 = sides.get("cathetus1")
    c2 = sides.get("cathetus2")
    h = sides.get("hypotenuse")

    # Calculate the missing side using the Pythagorean theorem
    if c1 is None:
        c1 = (h**2 - c2**2) ** 0.5
    elif c2 is None:
        c2 = (h**2 - c1**2) ** 0.5
    # NOTE: We do not look at `h` missing as it's not needed for printing.

    # Determine the size for printing the triangle
    height = int(round(c1))
    width = int(round(c2))
    ratio = width / height

    # Print the triangle
    for x in range(1, height + 1):
        n_stars = int(round(ratio * x))
        n_dots = width - n_stars
        stars = n_stars * "*"
        dots = n_dots * "."
        print(dots + stars)


def machine(commands: list[int]):
    row = ""
    for command in commands:
        if command == 0:
            row += "."
        elif command == 1:
            row += "#"
        elif command == 2:
            print(row)
            row = ""
    print(row, end="")


class City(BaseModel):
    name: str
    pop: int


def smallest_city(cities: list[City]) -> City:
    smallest = cities[0]
    for city in cities:
        if city.pop < smallest.pop:
            smallest = city
    return smallest
