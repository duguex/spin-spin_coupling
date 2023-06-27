from multiprocessing import Pool


def square(num):
    result = num ** 2
    print(f"result is {result}")
    return result


if __name__ == "__main__":
    p = Pool(5)
    p.map(square, [1, 2, 3])
