import json

# Step 2: Create dictionary
student = {
    "name": "Chaya",
    "age": 22,
    "course": "MCA",
    "marks": 85
}

# Step 3: Access values
print("Name:", student["name"])
print("Marks:", student["marks"])

# Step 4: Update and delete
student["marks"] = 90
del student["age"]

# Step 5: Loop dictionary
print("\nStudent Dictionary:")
for key, value in student.items():
    print(key, ":", value)

# Step 6 & 7: Convert to JSON and save file
with open("student.json", "w") as f:
    json.dump(student, f, indent=4)

print("\nData saved to student.json")

# Step 8: Read JSON file
with open("student.json", "r") as f:
    data = json.load(f)

# Step 9: Print formatted output
print("\nStudent Details From JSON:")
for k, v in data.items():
    print(k, ":", v)
