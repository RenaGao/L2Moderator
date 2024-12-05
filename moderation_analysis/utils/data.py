import json

def extract_text_in_parentheses(s):
    start = s.find('(')
    end = s.find(')', start)
    if start != -1 and end != -1:
        return s[start + 1:end].strip()
    else:
        return s.strip()


def merge_short_strings(strings):
    result = []
    i = 0

    while i < len(strings):
        current = strings[i].strip()  # Remove leading/trailing spaces
        current_tokens = current.split()

        # Keep merging while current string has less than 3 tokens and there's a next string
        while len(current_tokens) < 4 and i + 1 < len(strings):
            i += 1
            next_string = strings[i].strip()
            current += " " + next_string
            current_tokens = current.split()

        result.append(current)
        i += 1  # Move to the next string

    return result


def write_txt(lines:list, filepath:str):
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def load_json_data(path):
    with open(path) as f:
        json_objs = json.load(f)
        return json_objs