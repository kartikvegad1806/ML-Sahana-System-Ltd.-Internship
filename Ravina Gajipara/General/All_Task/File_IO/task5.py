def write_data(filepath):
        # Step 1: Take number of lines
        n = int(input("Enter number of lines: "))

        # Step 2: Write data to file
        with open(filepath, "a") as f:
            for i in range(n):
                line = input(f"Enter line {i+1}: ")
                f.write(line + "\n")
        print("\nData written successfully!\n")

def read_data(filepath):
        # Step 3: Read data from file 
        with open(filepath, "r") as f:
            content = f.read()
        
        print("File Content:\n")
        print(content)

         # Step 4: Counting
        words = len(content.split())
        lines = len(content.splitlines())
        chars_with_space = len(content)
        chars_without_space = len(content.replace(" ", "").replace("\n", ""))
    
        print("Total Lines:", lines)
        print("Total Words:", words)
        print("Characters (with spaces):", chars_with_space)
        print("Characters (without spaces):", chars_without_space)

# User passes filename + location
path = input("Enter full file path (example: D:\\test.txt): ")

write_data(path)
read_data(path)
