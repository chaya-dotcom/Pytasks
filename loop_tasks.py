# 1. Print numbers 1 to 100
for i in range(1, 101):
    print(i)

# 2. Countdown using while loop
count = 5
while count > 0:
    print("Countdown:", count)
    count -= 1

# 3. Break example
for i in range(1, 11):
    if i == 6:
        break
    print(i)

# 4. Continue example
for i in range(1, 6):
    if i == 3:
        continue
    print(i)

# 5. Iterate over string
name = "Python"
for char in name:
    print(char)

# 6. Multiplication table
num = 5
for i in range(1, 11):
    print(num, "x", i, "=", num * i)

# 7. Range with step
for i in range(2, 21, 2):
    print(i)

# 8. Loop with conditions
for i in range(1, 11):
    if i % 2 == 0:
        print(i, "is Even")
    else:
        print(i, "is Odd")
