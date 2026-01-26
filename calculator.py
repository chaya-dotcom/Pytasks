# calculator.py

def add(a, b=0):
    """Returns sum of two numbers"""
    return a + b


def subtract(a, b):
    """Returns difference of two numbers"""
    return a - b


def multiply(a, b):
    """Returns product of two numbers"""
    return a * b


def divide(a, b):
    """Returns division result"""
    if b == 0:
        return "Cannot divide by zero"
    return a / b


def main():
    print("----- Simple Calculator -----")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    choice = input("Enter choice (1/2/3/4): ")

    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    if choice == '1':
        print("Result:", add(num1, num2))

    elif choice == '2':
        print("Result:", subtract(num1, num2))

    elif choice == '3':
        print("Result:", multiply(num1, num2))

    elif choice == '4':
        print("Result:", divide(num1, num2))

    else:
        print("Invalid input")


if __name__ == "__main__":
    main()2
    
