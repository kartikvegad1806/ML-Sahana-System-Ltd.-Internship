def write_data(n):
    students = []
    
    # Get Input for student information
    for i in range(n):
        rollno = input(f"Enter roll number for student {i+1}: ")
        name = input(f"Enter name for student {i+1}: ")
        students.append({'rollno': rollno, 'name': name})
    
    # Get Input of marks for each student
    for i in range(n):
        print(f"Enter marks for student {students[i]['rollno']} - {students[i]['name']}")
        mark1 = int(input("Enter marks for subject 1: "))
        mark2 = int(input("Enter marks for subject 2: "))
        mark3 = int(input("Enter marks for subject 3: "))
        students[i]['marks'] = [mark1, mark2, mark3]
        students[i]['average'] = (mark1 + mark2 + mark3) / 3
    
    # Writing the student info into studentInfo.txt
    with open('studentInfo.txt', 'w') as f:
        for student in students:
            f.write(f"{student['rollno']}-{student['name']}\n")
    
    # Writing the marks of student into studentMarks.txt
    with open('studentMarks.txt', 'w') as f:
        for student in students:
            marks_str = '-'.join(map(str, student['marks']))
            f.write(f"{student['rollno']}-{marks_str}\n")
    
    # Sorting the students by average in descending order
    students.sort(key=lambda x: x['average'], reverse=True)
    
    # Categorizing students by grade and writing them to their respective files
    a_grade_students = []
    b_grade_students = []
    c_grade_students = []
    
    for student in students:
        avg = student['average']
        if 80 <= avg <= 100:
            a_grade_students.append(student)
        elif 60 <= avg < 80:
            b_grade_students.append(student)
        elif 40 <= avg < 60:
            c_grade_students.append(student)
    
    # Writing A grade students to Agrade.txt file
    with open('Agrade.txt', 'w') as f:
        for student in a_grade_students:
            f.write(f"{student['rollno']}-{student['name']}-{student['average']:.2f}\n")
    
    # Writing B grade students to Bgrade.txt file
    with open('Bgrade.txt', 'w') as f:
        for student in b_grade_students:
            f.write(f"{student['rollno']}-{student['name']}-{student['average']:.2f}\n")
    
    # Writing C grade students to Cgrade.txt file
    with open('Cgrade.txt', 'w') as f:
        for student in c_grade_students:
            f.write(f"{student['rollno']}-{student['name']}-{student['average']:.2f}\n")

