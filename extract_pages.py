import os

with open('index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

pages_info = [
    ('dashboard', 89, 182),
    ('lexicon', 185, 291),
    ('dataset', 294, 337),
    ('preprocess', 340, 398),
    ('batch', 401, 436),
    ('tfidf-config', 439, 522),
    ('tfidf-vocab', 525, 562),
    ('tfidf-matrix', 565, 603),
    ('tfidf-single', 606, 657),
    ('theory', 688, 793)
]

for page, start, end in pages_info:
    # Create directory
    os.makedirs(f'Pages/{page}', exist_ok=True)
    
    # Extract inner content (1-indexed start/end means 0-indexed start-1 and end-1)
    inner_content = lines[start:end-1]
    
    # Save to file
    with open(f'Pages/{page}/{page}.html', 'w', encoding='utf-8') as f:
        f.writelines(inner_content)
    
    # Replace the lines in `lines` with empty content, but keeping the wrapper
    # Since we shouldn't mess up line numbers during iteration, we replace the inner lines with empty strings,
    # except one line if needed, or just empty strings
    for i in range(start, end-1):
        lines[i] = ""

# Write back to index.html
with open('index.html', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Extraction complete.")
