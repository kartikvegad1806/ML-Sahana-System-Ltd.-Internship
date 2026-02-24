def write_data(n):
    students = []
    marks = []
    
    print("Enter student details:")
    for i in range(n):
        rollno = input(f"Enter roll number for student {i+1}: ")
        name = input(f"Enter name for student {i+1}: ")
        students.append((rollno, name))
    
    print("\nEnter marks for 3 subjects:")
    for i in range(n):
        print(f"Student {i+1} ({students[i][0]}):")
        mark1 = input("Subject 1 marks: ")
        mark2 = input("Subject 2 marks: ")
        mark3 = input("Subject 3 marks: ")
        marks.append((students[i][0], int(mark1), int(mark2), int(mark3)))
    
    with open('studentInfo.txt', 'w') as f:
        for rollno, name in students:
            f.write(f"{rollno}-{name}\n")
    
    with open('studentMarks.txt', 'w') as f:
        for rollno, m1, m2, m3 in marks:
            f.write(f"{rollno}-{m1}-{m2}-{m3}\n")
    
    student_grades = []
    for rollno, m1, m2, m3 in marks:
        avg = (m1 + m2 + m3) / 3
        name = next(name for r, name in students if r == rollno)
        student_grades.append((rollno, name, avg))
    
    student_grades.sort(key=lambda x: x[2], reverse=True)
    
    a_grade = [s for s in student_grades if s[2] >= 80]
    b_grade = [s for s in student_grades if 60 <= s[2] < 80]
    c_grade = [s for s in student_grades if 40 <= s[2] < 60]
    
    with open('Agrade.txt', 'w') as f:
        for rollno, name, avg in a_grade:
            f.write(f"{rollno}-{name}-{avg:.2f}\n")
    
    with open('Bgrade.txt', 'w') as f:
        for rollno, name, avg in b_grade:
            f.write(f"{rollno}-{name}-{avg:.2f}\n")
    
    with open('Cgrade.txt', 'w') as f:
        for rollno, name, avg in c_grade:
            f.write(f"{rollno}-{name}-{avg:.2f}\n")
    
    print("Data stored successfully!")

n = int(input("Enter number of students: "))
write_data(n)