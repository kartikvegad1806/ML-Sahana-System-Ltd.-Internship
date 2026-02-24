from function2 import write_student_info, write_student_marks, calculate_average, bifurcate_grades

n = int(input("Enter number of students: "))

write_student_info(n)
write_student_marks(n)
calculate_average()
bifurcate_grades()

print("Data processed successfully.")
