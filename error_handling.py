import logging
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
try:
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))
    result = a / b
    print(a / b)
    print("Result:", result)
except ZeroDivisionError:
    logging.error("Division by zero")
    print("Cannot divide by zero")

except ValueError:
    logging.error("Invalid input")
    print("Enter numbers only")
else:
    print("Calculation successful")
finally:
    print("Program finished")
