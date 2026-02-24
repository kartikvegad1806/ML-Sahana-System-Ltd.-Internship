import sys

def write_to_file(filename, lines):
    with open(filename, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    print(f"Data written to {filename}")

def read_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def reverse_write_to_file(filename, lines):
    with open(filename, 'w') as file:
        for line in reversed(lines):
            file.write(line + '\n')
    print(f"Reversed data written to {filename}")

def replace_word_in_file(filename, old_word, new_word):
    with open(filename, 'r') as file:
        content = file.read()
    
    new_content = content.replace(old_word, new_word)
    
    with open(filename, 'w') as file:
        file.write(new_content)
    print(f"Replaced '{old_word}' with '{new_word}' in {filename}")

def main():
    file_location = input("Enter file location: ").strip()
    demo_file = file_location + "/demo.txt"
    dummy_file = file_location + "/dummy.txt"
    
    num_lines = int(input("Enter number of lines: "))
    lines = []
    for i in range(num_lines):
        line = input(f"Enter line {i+1}: ")
        lines.append(line)
    
    write_to_file(demo_file, lines)
    
    read_lines = read_from_file(demo_file)
    print(f"Data from {demo_file}: {read_lines}")
    
    reverse_write_to_file(dummy_file, read_lines)
    
    old_word = input("Enter word to replace: ").strip()
    new_word = input("Enter replacement word: ").strip()
    replace_word_in_file(dummy_file, old_word, new_word)

if __name__ == "__main__":
    main()