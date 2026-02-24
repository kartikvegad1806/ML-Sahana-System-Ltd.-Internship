def calculate(*lists):

    if len(lists) == 1:
        print(lists[0])
    
    elif len(lists) == 2:
        concatenated = lists[0] + lists[1]
        print(f"Max: {max(concatenated)}, Min: {min(concatenated)}")
    
    elif len(lists) == 3:
        concatenated = lists[0] + lists[1] + lists[2]
        print(f"Sum of all elements: {sum(concatenated)}")
    
    else:
        concatenated = []
        for lst in lists:
            concatenated.extend(lst)
        
        squared = list(map(lambda x: x**2, concatenated))
        print(f"Squared elements: {squared}")
        
        odd_numbers = list(filter(lambda x: x % 2 != 0, concatenated))
        print(f"Odd numbers: {odd_numbers}")


num_lists = int(input("Enter number of lists: "))
lists = []

for i in range(num_lists):
    n = int(input(f"Enter number of elements for list {i+1}: "))
    elements = list(map(int, input(f"Enter elements for list {i+1}: ").split()))
    lists.append(elements)

calculate(*lists)
