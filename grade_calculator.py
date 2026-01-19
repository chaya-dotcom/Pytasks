# Step 1: Take marks input from the user
marks = int(input("Enter your marks (0 - 100): "))

# Step 2: Check for invalid marks
if marks < 0 or marks > 100:
    print("❌ Invalid marks! Please enter marks between 0 and 100.")

# Step 3: Calculate grade using conditions
elif marks >= 90 and marks <= 100:
    print("Grade: A")
    
    # Step 4: Nested condition (business rule)
    if marks >= 95:
        print("🏆 Distinction")

elif marks >= 75 and marks < 90:
    print("Grade: B")
    print("👍 Very Good Performance")

elif marks >= 60 and marks < 75:
    print("Grade: C")
    print("🙂 Good Performance")

elif marks >= 35 and marks < 60:
    print("Grade: D")
    print("⚠ Needs Improvement")

else:
    print("❌ Fail")
