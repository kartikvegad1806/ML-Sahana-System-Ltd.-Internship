ls = []
n = int(input("Enter number of elements: "))


for i in range(n):
    element = int(input(f"Enter element {i+1}: "))
    ls.append(element)

print(f"Main list: {ls}")
print(f"Total: {sum(ls)}")

def divide_list(lst):
    total_sum = sum(lst)
    
    if total_sum % 2 != 0:
        print("Equal division not possible (odd sum)")
        return
    
    target_sum = total_sum // 2
    ls1 = []
    ls2 = []
    current_sum = 0
    
    for element in lst:
        if current_sum + element <= target_sum:
            ls1.append(element)
            current_sum += element
        else:
            ls2.append(element)
    
    # Check if sums are equal
    if sum(ls1) == sum(ls2):
        print(f"\nls1: {ls1} => {sum(ls1)}")
        print(f"ls2: {ls2} => {sum(ls2)}")
        print("Equal division successful!")
    else:
        print(f"ls1: {ls1} => {sum(ls1)}")
        print(f"ls2: {ls2} => {sum(ls2)}")
        print("Equal division not possible with this arrangement")

divide_list(ls)