# Task 6: Lists, Tuples & Sets

print("------ COLLECTIONS DEMO ------\n")

# STEP 1: Create a list of students
students = ["Chaya", "Anu", "Ravi", "Chaya"]
print("Original Student List:", students)

# STEP 2: Add a student
students.append("Kiran")
print("\nAfter Adding Kiran:", students)

# STEP 3: Remove a student
students.remove("Ravi")
print("\nAfter Removing Ravi:", students)

# STEP 4: Sort students
students.sort()
print("\nAfter Sorting:", students)

# STEP 5: Tuple for fixed data
college_info = ("ABC College", 2026)
print("\nCollege Info (Tuple):", college_info)

# STEP 6: Convert list to set to remove duplicates
unique_students = set(students)
print("\nUnique Students (Set):", unique_students)

# STEP 7: Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}

print("\nSet Union:", set1.union(set2))
print("Set Intersection:", set1.intersection(set2))

# STEP 8: Iteration over list
print("\nIterating Students:")
for name in students:
    print(name)

# STEP 9: Mutable vs Immutable

# List is mutable
students[0] = "NewName"
print("\nModified List:", students)

# Tuple is immutable (cannot change)
print("\nTuple remains unchanged:", college_info)

# STEP 10: Formatted Output
print(f"\nFinal Students List: {students}")
print("\n------ TASK COMPLETED ------")

