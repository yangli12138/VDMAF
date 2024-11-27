import json


with open('../dataset/function', 'r') as infile:
    data = json.load(infile)
code = ""
progress = 0
for r in data:
    progress = progress + 1
    code = code + r["func"]

print(progress)
with open('corpus', 'w', encoding='utf-8') as f:
    f.write(code)