import math
from typing import Optional

from pydantic.dataclasses import dataclass

from benchmarks.utils import Print


def square(size: int) -> Print:
    for x in range(size):
        if (x == 0) or (x == size - 1):
            print("*" * size)
        else:
            print("*" + ("." * (size - 2)) + "*")


def triangle(size: int) -> Print:
    for x in range(1, size + 1):
        n_stars = x
        n_dots = size - n_stars
        stars = n_stars * "*"
        dots = n_dots * "."
        print(dots + stars)


def rhombus(size: int) -> Print:
    limit = (size + 1) // 2
    for start, end, step in [(0, limit, 1), (limit - 1 - (size % 2), -1, -1)]:
        for x in range(start, end, step):
            stars = 1 + (x * 2)
            dots = (size - stars) // 2
            row = "." * dots + "*" * stars + "." * dots
            print(row)


def histogram(nums: list[int]) -> Print:
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


def scatter(size: int, points: list[tuple[int, int]]) -> Print:
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


def generate_trajectory(velocity: int, angle: int, size: list[int]) -> Print:
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


def smallest_number(x: int, y: int, z: int) -> int:
    return min(x, y, z)


def complementary_dna(sequence: list[str]) -> list[str]:
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return [complement[nucleotide] for nucleotide in sequence]


def calculate_discount(shopping: list[float], discount: int) -> float:
    total = 0
    for item in shopping:
        total += item * (1 - discount / 100)
    return total


def even_numbers(numbers: list[int]) -> list[int]:
    return [number for number in numbers if number % 2 == 0]


def city_life(turin: set[str], milan: set[str]) -> set[str]:
    return turin.intersection(milan)


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


def pascal_triangle(n: int) -> list[list[int]]:
    triangle = [[1]]
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        row.append(1)
        triangle.append(row)
    return triangle


def missing_triangle(sides: dict) -> Print:
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


def machine(commands: list[int]) -> Print:
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


@dataclass
class City:
    name: str
    population: int


def smallest_city(cities: list[City]) -> City:
    smallest = cities[0]
    for city in cities:
        if city.population < smallest.population:
            smallest = city
    return smallest


@dataclass
class Person:
    name: str
    height: int


def total_height(person_a: Person, person_b: Person) -> int:
    return person_a.height + person_b.height


@dataclass
class Book:
    title: str
    pages: int

    def __eq__(self, other):
        return self.title == other.title


def total_num_pages(books: list[Book]) -> int:
    total_pages = 0
    for book in books:
        total_pages += book.pages
    return total_pages


@dataclass
class Car:
    license_plate: str
    is_parked: bool


def park_cars(garage: list[Car], license_plates: list[str]) -> int:
    parked_count = 0
    for car in garage:
        if car.license_plate in license_plates and not car.is_parked:
            car.is_parked = True
            parked_count += 1
    return parked_count


@dataclass
class Student:
    name: str
    grades: list[float]


def calculate_gpa(students: list[Student], name: str) -> float:
    for student in students:
        if student.name == name:
            total_grades = sum(student.grades)
            num_subjects = len(student.grades)
            if num_subjects == 0:  # To prevent division by zero
                return 0
            return total_grades / num_subjects
    return 0  # If student not found


@dataclass
class Coordinate:
    x: int
    y: int

    def distance(self, other: "Coordinate") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Warehouse:
    products: int

    def add_products(self, amount: int):
        self.products += amount

    def remove_products(self, amount: int):
        self.products -= min(amount, self.products)

    def get_products(self) -> int:
        return self.products


@dataclass
class BookStore:
    books: list[Book]

    def add_book(self, book: Book):
        self.books.append(book)

    def get_num_copies(self, book: Book) -> int:
        count = 0
        for b in self.books:
            if b.title == book.title:
                count += 1
        return count


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Obstacle:
    position: Position


@dataclass
class SimpleCar:
    position: Position
    obstacles: list[Obstacle]

    def move_up(self):
        new_position = Position(x=self.position.x, y=self.position.y + 1)
        if not self._is_obstacle(new_position):
            self.position = new_position

    def move_down(self):
        new_position = Position(x=self.position.x, y=self.position.y - 1)
        if not self._is_obstacle(new_position):
            self.position = new_position

    def move_left(self):
        new_position = Position(x=self.position.x - 1, y=self.position.y)
        if not self._is_obstacle(new_position):
            self.position = new_position

    def move_right(self):
        new_position = Position(x=self.position.x + 1, y=self.position.y)
        if not self._is_obstacle(new_position):
            self.position = new_position

    def get_position(self) -> Position:
        return self.position

    def _is_obstacle(self, new_position: Position) -> bool:
        return any(obstacle.position == new_position for obstacle in self.obstacles)


CIPHER_MAPPING: dict[str, int] = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "i": 8,
    "j": 9,
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "q": 16,
    "r": 17,
    "s": 18,
    "t": 19,
    "u": 20,
    "v": 21,
    "w": 22,
    "x": 23,
    "y": 24,
    "z": 25,
    " ": 26,
    ",": 27,
    ".": 28,
    "!": 29,
    "?": 30,
}


@dataclass
class Cipher:
    key: int
    message: str

    def encrypt(self):
        # Convert message to numbers based on the alphabet
        vector = []
        for char in self.message:
            vector.append(CIPHER_MAPPING[char])

        # Calculate shift amount
        if self.key > len(CIPHER_MAPPING):
            shift = self.key % len(CIPHER_MAPPING)
        else:
            shift = self.key

        # Shift each number in the vector
        for i in range(len(vector)):
            vector[i] = (vector[i] + shift) % len(CIPHER_MAPPING)

        # Convert numbers back to letters
        letters = list(CIPHER_MAPPING.keys())
        encrypted_message = ""
        for num in vector:
            encrypted_message += letters[num]

        self.message = encrypted_message

    def decrypt(self):
        # Convert encrypted message to numbers
        vector = []
        for char in self.message:
            vector.append(CIPHER_MAPPING[char])

        # Calculate shift amount
        if self.key > len(CIPHER_MAPPING):
            shift = self.key % len(CIPHER_MAPPING)
        else:
            shift = self.key

        # Reverse shift for each number
        for i in range(len(vector)):
            vector[i] = (vector[i] - shift) % len(CIPHER_MAPPING)

        # Convert numbers back to letters
        letters = list(CIPHER_MAPPING.keys())
        decrypted_message = ""
        for num in vector:
            decrypted_message += letters[num]

        self.message = decrypted_message

    def get_message(self) -> str:
        return self.message


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Line:
    a: int
    b: int
    c: int

    def is_parallel(self, other_line: "Line") -> bool:
        return self.a * other_line.b == self.b * other_line.a

    def intersection(self, other_line: "Line") -> Optional[Point]:
        if self.is_parallel(other_line):
            return None

        x = (self.c * other_line.b - self.b * other_line.c) / (
            self.a * other_line.b - self.b * other_line.a
        )
        y = (self.a * other_line.c - self.c * other_line.a) / (
            self.a * other_line.b - self.b * other_line.a
        )

        return Point(x, y)

    def contains_point(self, point: Point) -> bool:
        return self.a * point.x + self.b * point.y == self.c


@dataclass
class FairDice:
    face: int

    def roll(self) -> int:
        self.face = (self.face + 5) % 6
        return self.face


@dataclass
class Player:
    name: str
    score: int

    def roll_and_update(self, dice: FairDice):
        self.score += dice.roll()


@dataclass
class DiceGame:
    players: list[Player]
    dice: FairDice

    def play(self):
        for player in self.players:
            player.roll_and_update(self.dice)

    def get_winner(self) -> str:
        winner = self.players[0]
        for player in self.players:
            if player.score > winner.score:
                winner = player

        for player in self.players:
            player.score = 0

        return winner.name


@dataclass
class Candidate:
    name: str
    hair_color: str
    eye_color: str
    has_glasses: bool

    def __hash__(self):
        return hash((self.hair_color, self.eye_color, self.has_glasses))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        return False


@dataclass
class GuessWhoGame:
    answer: Candidate
    candidates: list[Candidate]

    def guess(self, candidate: Candidate) -> bool:
        return (
            candidate.name == self.answer.name
            and candidate.eye_color == self.answer.eye_color
            and candidate.hair_color == self.answer.hair_color
            and candidate.has_glasses == self.answer.has_glasses
        )

    def ask_hair_color(self, hair_color: str) -> bool:
        return self.answer.hair_color == hair_color

    def ask_eye_color(self, eye_color: str) -> bool:
        return self.answer.eye_color == eye_color

    def ask_glasses(self) -> bool:
        return self.answer.has_glasses


@dataclass
class Register:
    name: str
    prev: Optional["Register"] = None
    next: Optional["Register"] = None

    def add(self, name: str):
        if self.name > name:
            if self.prev is None:
                self.prev = Register(name=name, next=self)
            else:
                self.next = Register(name=self.name, prev=self, next=self.next)
                self.name = name
        else:
            if self.next is None:
                self.next = Register(name=name, prev=self)
            else:
                self.next.add(name)

    def remove(self, name: str):
        if self.name == name:
            if self.prev is not None:
                self.prev.next = self.next
            if self.next is not None:
                self.next.prev = self.prev
        else:
            if self.next is not None:
                self.next.remove(name)

    def get_names(self) -> list[str]:
        if self.next is None:
            return [self.name]
        else:
            return [self.name] + self.next.get_names()


@dataclass
class Queue:
    name: str
    next: Optional["Queue"] = None

    def add(self, name: str):
        if self.next is None:
            self.next = Queue(name=name)
        else:
            self.next.add(name)

    def remove(self) -> str:
        if self.next is not None:
            removed_name = self.name
            self.name = self.next.name
            self.next = self.next.next
            return removed_name

    def get_names(self) -> list[str]:
        if self.next is None:
            return [self.name]
        else:
            return [self.name] + self.next.get_names()


@dataclass
class Stack:
    parenthesis: str
    prev: Optional["Stack"] = None

    def add(self, parenthesis: str):
        next_parenthesis = self.parenthesis
        self.parenthesis = parenthesis
        self.prev = Stack(parenthesis=next_parenthesis, prev=self)

    def match(self, parenthesis: str) -> bool:
        cur_parenthesis = self.parenthesis
        if cur_parenthesis == "(" and parenthesis == ")":
            if self.prev is not None:
                self.parenthesis = self.prev.parenthesis
                self.prev = self.prev.prev
            return True
        return False
