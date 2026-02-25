def calculate(*lists):

    # Case 1 → Only one list
    if len(lists) == 1:
        print("List:", lists[0])

    # Case 2 → Two lists
    elif len(lists) == 2:
        combined = lists[0] + lists[1]
        print("Concatenated List:", combined)
        print("Maximum:", max(combined))
        print("Minimum:", min(combined))

    # Case 3 → Three lists
    elif len(lists) == 3:
        combined = []
        for lst in lists:
            combined += lst

        print("Concatenated List:", combined)
        print("Total Sum:", sum(combined))

    # Case 4 → More than 3 lists
    else:
        combined = []
        for lst in lists:
            combined += lst

        print("Concatenated List:", combined)

        square_list = list(map(lambda x: x*x, combined))
        print("Square List:", square_list)

        odd_list = list(filter(lambda x: x % 2 != 0, combined))
        print("Odd List:", odd_list)


# User Input Section

num_lists = int(input("Enter number of lists: "))

if num_lists == 1:
    n = int(input(f"Enter number of elements for list 1: "))
    ls1 = []

    for i in range(n):
        element = int(input(f"Enter element {i+1}: "))
        ls1.append(element)

    calculate(ls1)

elif num_lists == 2:
    ls1 = []
    ls2 = []

    n1 = int(input(f"Enter number of elements for list 1: "))
    for i in range(n1):
        element = int(input(f"Enter element {i+1}: "))
        ls1.append(element)

    n2 = int(input(f"Enter number of elements for list 2: "))
    for i in range(n2):
        element = int(input(f"Enter element {i+1}: "))
        ls2.append(element)

    calculate(ls1, ls2)

elif num_lists == 3:
    ls1 = []
    ls2 = []
    ls3 = []

    n1 = int(input(f"Enter number of elements for list 1: "))
    for i in range(n1):
        element = int(input(f"Enter element {i+1}: "))
        ls1.append(element)

    n2 = int(input(f"Enter number of elements for list 2: "))
    for i in range(n2): 
        element = int(input(f"Enter element {i+1}: "))
        ls2.append(element)

    n3 = int(input(f"Enter number of elements for list 3: "))
    for i in range(n3):
        element = int(input(f"Enter element {i+1}: "))
        ls3.append(element)

    calculate(ls1, ls2, ls3)

else:
    all_lists = []

    for i in range(1, num_lists + 1):
        n = int(input(f"Enter number of elements for list {i}: "))
    
        temp_list = []
        for j in range(n):
            element = int(input(f"Enter element {j+1}: "))
            temp_list.append(element)
    
        all_lists.append(temp_list)
        
        calculate(*all_lists)

