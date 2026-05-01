import sys
with open(r"D:\OneDrive\Desktop\train\build_dataset.py", 'rb') as f:
    data = f.read()

# Find "层间弱" in bytes
text = data.decode('utf-8')
lines = text.split('\n')
for i, line in enumerate(lines):
    if '层间弱' in line:
        # Find the position
        pos = line.index('层间弱')
        # Show bytes before and after
        snippet = line[pos-30:pos+30]
        print(f"Line {i+1}")
        for ch in snippet:
            cp = ord(ch)
            if cp > 127:
                print(f"  U+{cp:04X} ({ch})")
            else:
                print(f"  U+{cp:04X} ({ch!r})")
