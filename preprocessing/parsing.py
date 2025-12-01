import re

def split_numbered_items(block, item_pattern):
    lines = block.strip().split("\n")
    items = []
    current = []

    for line in lines:
        if item_pattern.match(line):
            if current:
                items.append("\n".join(current).strip())
            current = [line]
        else:
            if current:
                current.append(line)
            else:
                current = [line]
    if current:
        items.append("\n".join(current).strip())

    return items

def parse(text):

    item_pattern = re.compile(r"^(\d{1,3}[A-Z]{0,3})[\.]?\s+")
    marker_pat = re.compile(r"^(_title_|_subtitle_|_subsection_|_subsubsection_)(.*)$")

    title_doc = re.findall("_doc_title_(.*)", text)[0]
    text = re.sub("_doc_title_.*\n","",text)
    
    current_title = None
    current_subtitle = None
    current_subsection = None
    current_subsub = None

    final = []
    lines = text.split("\n")

    buffer = []

    def flush_buffer():
        if buffer:
            block = "\n".join(buffer).strip()
            items = split_numbered_items(block,item_pattern)
            for it in items:
                final.append({
                    "doc_title": title_doc,          # <-- NEW FIELD
                    "title": current_title,
                    "subtitle": current_subtitle,
                    "subsection": current_subsection,
                    "subsubsection": current_subsub,
                    "content": it
                })
        buffer.clear()

    for line in lines:
        m = marker_pat.match(line)
        if m:
            flush_buffer()
            marker, label = m.group(1), m.group(2).strip()
            if marker == "_title_":
                current_title = label
                current_subtitle = None
                current_subsection = None
                current_subsub = None
            elif marker == "_subtitle_":
                current_subtitle = label
                current_subsection = None
                current_subsub = None
            elif marker == "_subsection_":
                current_subsection = label
                current_subsub = None
            elif marker == "_subsubsection_":
                current_subsub = label
            continue

        buffer.append(line)

    flush_buffer()
    return final
