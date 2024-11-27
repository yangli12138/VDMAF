import json

with open('../dataset/vulnerables.json', 'r') as v_file:
    vulnerables = json.load(v_file)

with open('../dataset/non-vulnerables.json', 'r') as nv_file:
    non_vulnerables = json.load(nv_file)

data = vulnerables + non_vulnerables
code = ""
progress = 0
for r in data:
    if len(r["code"]) <= 200:
        progress = progress + 1
        if progress == 40000:
            break
        code = code + r["code"]

print(progress)

with open('corpus', 'w', encoding='utf-8') as f:
    f.write(code)