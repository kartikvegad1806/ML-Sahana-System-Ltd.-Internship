def write_student_info(n):
    with open("studentinfo.txt", "w") as f:
        for _ in range(n):
            roll = input("Enter roll number: ")
            name = input("Enter name: ")
            f.write(f"{roll}-{name}\n")


def write_student_marks(n):
    subjects = int(input("Enter number of subjects: "))
    with open("studentmarks.txt", "w") as f:
        for _ in range(n):
            roll = input("Enter roll number: ")
            marks = []
            for i in range(subjects):
                mark = float(input(f"Enter marks of subject {i+1}: "))
                marks.append(str(mark))
            line = roll + "-" + "-".join(marks)
            f.write(line + "\n")


def calculate_average():
    marks_data = {}
    with open("studentmarks.txt", "r") as f:
        for line in f:
            parts = line.strip().split("-")
            roll = parts[0]
            marks = list(map(float, parts[1:]))
            avg = sum(marks) / len(marks)
            marks_data[roll] = avg

    updated_lines = []
    with open("studentinfo.txt", "r") as f:
        for line in f:
            parts = line.strip().split("-")
            roll = parts[0]
            name = parts[1]
            avg = marks_data.get(roll, 0)
            updated_lines.append(f"{roll}-{name}-{avg}")

    with open("studentinfo.txt", "w") as f:
        for line in updated_lines:
            f.write(line + "\n")


def bifurcate_grades():
    with open("studentinfo.txt", "r") as f, \
         open("Agrade.txt", "w") as a, \
         open("Bgrade.txt", "w") as b, \
         open("Cgrade.txt", "w") as c:

        for line in f:
            parts = line.strip().split("-")
            roll = parts[0]
            name = parts[1]
            avg = float(parts[2])

            if 80 < avg < 100:
                a.write(line)
            elif 60 < avg < 80:
                b.write(line)
            else:
                c.write(line)
