import os

files_to_fix = [
    'core/qwen_interface.py',
    'core/interfaces.py'
]

replacements = {
    '✓': '[OK]',
    '⚠️': '[!]',
    '⚠': '[!]'
}

base_path = 'c:/Users/Administrator/Desktop/stdpbrain'

for file_path in files_to_fix:
    full_path = os.path.join(base_path, file_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'Fixed: {file_path}')

print('Done!')
