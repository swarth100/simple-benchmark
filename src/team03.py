import time


def square(size: int):
    for x in range(size):
        if (x == 0) or (x == size - 1):
            print("*" * size)
        else:
            print("*" + ("." * (size - 2)) + "*")

    # Artificial delay to slow down implementation
    time.sleep(0.001)


def triangle(size: int):
    for x in range(1, size + 1):
        n_stars = x
        n_dots = size - n_stars
        stars = n_stars * "*"
        dots = n_dots * "."
        print(dots + stars)

    # Artificial delay to slow down implementation
    time.sleep(0.001)
