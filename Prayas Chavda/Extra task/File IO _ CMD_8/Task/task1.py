import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python task1.py <file_location> <filename>")
        sys.exit(1)
    
    file_location = sys.argv[1]
    filename = sys.argv[2]
    filepath = f"{file_location}/{filename}"
    
    num_lines = int(input("Enter number of lines: "))
    
    print(f"Enter {num_lines} lines of data:")
    with open(filepath, 'w') as file:
        for i in range(num_lines):
            line = input(f"Line {i+1}: ")
            file.write(line + '\n')
    
    with open(filepath, 'r') as file:
        content = file.read()
    
    lines = content.strip().split('\n')
    num_lines_count = len(lines)
    
    words = content.split()
    num_words = len(words)
    
    chars_with_space = len(content) - 1
    chars_without_space = len(content.replace(' ', '').replace('\n', '')) - 1
    
    print("\n--- File Analysis ---")
    print(f"Number of lines: {num_lines_count}")
    print(f"Number of words: {num_words}")
    print(f"Characters with spaces: {chars_with_space}")
    print(f"Characters without spaces: {chars_without_space}")

if __name__ == "__main__":
    main()