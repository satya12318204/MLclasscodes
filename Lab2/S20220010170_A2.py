import numpy as np

def compute_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * compute_factorial(n - 1)

def find_prime_factors(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)
    return factors

def apply_rotation_cipher(text, shift):
    result = ""
    for char in text:
        if char.isupper():
            result += chr((ord(char) - 65 + shift) % 26 + 65)
        elif char.islower():
            result += chr((ord(char) - 97 + shift) % 26 + 97)
        else:
            result += char
    return result

def calculate_letter_percentage(S, letter):
    count = S.count(letter)
    percentage = (count / len(S)) * 100
    return int(percentage)

def compute_distances(x, y, p):
    n = x.shape[0]
    euclidean_distances = []
    manhattan_distances = []
    minkowski_distances = []
    for i in range(n):
        sum_of_squares = sum((x[i][j] - y[i][j]) ** 2 for j in range(x.shape[1]))
        euclidean_distances.append(sum_of_squares ** 0.5)
        sum_of_absolute_differences = sum(abs(x[i][j] - y[i][j]) for j in range(x.shape[1]))
        manhattan_distances.append(sum_of_absolute_differences)
        sum_of_powers = sum(abs(x[i][j] - y[i][j]) ** p for j in range(x.shape[1]))
        minkowski_distances.append(sum_of_powers ** (1 / p))
    return euclidean_distances, manhattan_distances, minkowski_distances

def main():
    fact_result = compute_factorial(10)
    print(f"Factorial of given integer: {fact_result:,}")
    
    prime_factors_result = find_prime_factors(84)
    print(f"Prime factors of given number: {', '.join(map(str, prime_factors_result))}")
    
    text = "Satya Amrutha Raja Perisetty"
    shift = 1
    cipher_result = apply_rotation_cipher(text, shift)
    print(f"Rotation cipher of '{text}' with shift {shift}: {cipher_result}")
    
    S = "Satya Amrutha Raja Perisetty"
    letter = "a"
    percentage_result = calculate_letter_percentage(S, letter)
    print(f"Percentage of '{letter}' in '{S}': {percentage_result}%")
    
    n = 5
    x = np.random.rand(n, 2)
    y = np.random.rand(n, 2)
    
    print("\nGenerated 2D arrays:")
    print(f"x:\n{x}")
    print(f"y:\n{y}")
    
    p = 3
    euclidean_distances, manhattan_distances, minkowski_distances = compute_distances(x, y, p)
    
    print("\nEuclidean Distances between corresponding points in x and y:")
    for i, dist in enumerate(euclidean_distances, start=1):
        print(f"Point {i}: {dist:.4f}")

    print("\nManhattan Distances between corresponding points in x and y:")
    for i, dist in enumerate(manhattan_distances, start=1):
        print(f"Point {i}: {dist:.4f}")

    print(f"\nMinkowski Distances (p={p}) between corresponding points in x and y:")
    for i, dist in enumerate(minkowski_distances, start=1):
        print(f"Point {i}: {dist:.4f}")

if __name__ == "__main__":
    main()
