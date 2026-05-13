with open('index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'id="svm-c"' in line:
        start_del = i - 1
        end_del = start_del + 23
        lines[start_del:end_del] = ['<div id="page-classification" class="page"></div>\n']
        break

with open('index.html', 'w', encoding='utf-8') as f:
    f.writelines(lines)
