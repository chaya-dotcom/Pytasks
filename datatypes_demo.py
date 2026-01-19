
# STEP 1: Declare variables
age = 22                     # Integer
salary = 25000.50            # Float
name = "Chaya"               # String
is_student = True            # Boolean


# STEP 2: Print values and types
print("Value and Data Types:")
print(age, type(age))
print(salary, type(salary))
print(name, type(name))
print(is_student, type(is_student))
print("-" * 40)

# STEP 3: Arithmetic operations
future_age = age + 5
monthly_salary = salary / 12

print("Future Age after 5 years:", future_age)
print("Monthly Salary:", monthly_salary)
print("-" * 40)

# STEP 4: Type conversion (string to int & float)
try:
    num1 = input("Enter an integer number: ")
    num2 = input("Enter a decimal number: ")

    num1 = int(num1)          # Convert string to int
    num2 = float(num2)        # Convert string to float

    print("Addition Result:", num1 + num2)

except ValueError:
    print("Error: Please enter valid numeric values only.")
print("-" * 40)

# STEP 5: String and number concatenation
print("Name is:", name)
print("Age is:", age)
print("Age is " + str(age))   # Converting int to string
print("-" * 40)


# STEP 6: Dynamic typing demonstration
x = 10
print(x, type(x))

x = "Python Programming"
print(x, type(x))

x = 3.14
print(x, type(x))

print("-" * 40)

# STEP 7: Error handling example
try:
    value = int(input("Enter an integer value: "))
    print("You entered:", value)
except ValueError:
    print("Invalid input! Only integers are allowed.")

print("-" * 40)

print("Program executed successfully.")
