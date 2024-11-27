import json
with open('../dataset/function', 'r') as infile:
    data = json.load(infile)
code = ""
FFmpeg = 0
FFmpeg_1 = 0
FFmpeg_0 = 0
qemu = 0
qemu_1 = 0
qemu_0 = 0
for r in data:
    if r["project"] == "FFmpeg":
        FFmpeg += 1
        if r["target"] == 1:
            FFmpeg_1 += 1
        if r["target"] == 0:
            FFmpeg_0 += 1
    if r["project"] == "qemu":
        qemu += 1
        if r["target"] == 1:
            qemu_1 += 1
        if r["target"] == 0:
            qemu_0 += 1
print(FFmpeg)
print(FFmpeg_0)
print(FFmpeg_1)
print(qemu)
print(qemu_0)
print(qemu_1)