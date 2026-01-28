import csv

# ================= TXT FILE PART =================

print("---- TXT FILE OPERATIONS ----")

try:
    # Create + Write to TXT file
    with open("student.txt", "w") as file:
        file.write("Name: Chaya\n")
        file.write("Course: MCA\n")
        file.write("City: Shivamogga\n")

    print("TXT file created and data written.")

    # Read TXT file
    with open("student.txt", "r") as file:
        print("\nTXT File Content:")
        print(file.read())

    # Append data
    with open("student.txt", "a") as file:
        file.write("\nStatus: Completed")

    print("\nData appended successfully.")

except Exception as e:
    print("TXT File Error:", e)


# ================= CSV FILE PART =================

print("\n---- CSV FILE OPERATIONS ----")

try:
    # Create CSV and write header
    with open("students.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Age", "City"])

    # Append multiple rows
    with open("students.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Chaya", 22, "Shivamogga"])
        writer.writerow(["Anu", 21, "Bangalore"])
        writer.writerow(["Ravi", 23, "Mysore"])

    print("CSV file created and rows added.")

    # Read CSV file
    print("\nCSV File Content:")
    with open("students.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)

except Exception as e:
    print("CSV File Error:", e)


print("\nTask 8 Completed Successfully ✅")
