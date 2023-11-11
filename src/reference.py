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
