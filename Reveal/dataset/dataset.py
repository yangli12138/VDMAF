import json
from collections import Counter

# 读取 JSON 文件并统计条数和项目种类
def count_functions(vulnerables_path, non_vulnerables_path):
    with open(vulnerables_path, 'r') as v_file:
        vulnerables = json.load(v_file)

    with open(non_vulnerables_path, 'r') as nv_file:
        non_vulnerables = json.load(nv_file)

    # 统计数量
    total_functions = len(vulnerables) + len(non_vulnerables)
    positive_count = len(non_vulnerables)
    negative_count = len(vulnerables)

    # 统计项目种类
    project_positive_count = Counter(item['project'] for item in non_vulnerables)
    project_negative_count = Counter(item['project'] for item in vulnerables)

    # 组合项目统计
    project_counts = {project: {
        'Positive': project_positive_count[project],
        'Negative': project_negative_count[project]
    } for project in set(project_positive_count) | set(project_negative_count)}

    return total_functions, positive_count, negative_count, project_counts


# 文件路径
vulnerables_path = '../dataset/vulnerables.json'
non_vulnerables_path = '../dataset/non-vulnerables.json'

# 统计结果
total, positive, negative, project_counts = count_functions(vulnerables_path, non_vulnerables_path)

print(f'Functions: {total}')
print(f'Positive: {positive}')
print(f'Negative: {negative}')
print('Project Counts:')
for project, counts in project_counts.items():
    print(f'  {project}: Positive: {counts["Positive"]}, Negative: {counts["Negative"]}')
