
import os, re

def extract_title_and_creators(filename):
    filename = os.path.join(os.getcwd(), filename)
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expressions to match title and creator(s)
    title_pattern = re.compile(r'Title:\s*(.+?)(?=Creator)', re.MULTILINE | re.DOTALL)
    creator_pattern = re.compile(r'Creator\(s\):\s*(.+?)(?=Print Basis:)', re.MULTILINE | re.DOTALL)

    # Regular expression to find instances of "Image of page 1" or "Image of page 2" etc.
    image_pattern = re.compile(r'Image of page \d+')

    # If content has more than 3 instances of "Image of page 1" or "Image of page 2" etc., then it is not a book
    if len(image_pattern.findall(content)) > 3:
        return '', '', False
    else:
        # Extract title and creators from the content
        title_match = title_pattern.search(content)
        title = title_match.group(1).strip() if title_match else ''
        title = [title.strip() for title in title.split('\n') if title.strip()]
        title = ' '.join(title)

        creator_match = creator_pattern.search(content)
        creators = creator_match.group(1).strip() if creator_match else ''
        creators = [creator.strip() for creator in creators.split('\n') if creator.strip()]
        creators = ': '.join(creators)

        return title, creators, True



list_of_files = os.listdir('ccel')
print(f'\nWe have {len(list_of_files)} files in the library')

print('\nLoading documents...')

for file in list_of_files:
    try:
        title, creators, validation = extract_title_and_creators(f'ccel\\{file}')
    except Exception as e:
        print(f'Error in {file}')
    