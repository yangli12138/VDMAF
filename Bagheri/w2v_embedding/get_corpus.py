import json

mode = "sql"
# mode = ["sql", "xsrf", "xss", "command_injection", "open_redirect", "path_disclosure", "remote_code_execution"]

with open('../dataset/plain_'+mode, 'r') as infile:
    data = json.load(infile)

code = ""
progress = 0
for r in data:
    progress = progress + 1
    for c in data[r]:
        if "files" in data[r][c]:
            for f in data[r][c]["files"]:
                if not "source" in data[r][c]["files"][f]:
                    continue
                if "source" in data[r][c]["files"][f]:
                    sourcecode = data[r][c]["files"][f]["source"]
                    code = code + sourcecode
with open('corpus/corpus_withString' + mode, 'w', encoding='utf-8') as f:
    f.write(code)