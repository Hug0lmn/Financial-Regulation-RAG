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

def parse(text, name, appendix = False):

    #Item pattern detect a new mini paragraph
    item_pattern = re.compile(r"^([A-Z]?\d+[A-Z]?\.?\d*\.?\d*)\s+") #Lot of optional but need to take into account 13D / 3.2.1 / 2.3 / B.3.1 etc...
    marker_pat = re.compile(r"^(_title_|_subtitle_|_subsection_|_subsubsection_)(.*)$")

    text = re.sub("_doc_title_.*\n","",text)
    source = name

    if appendix :
        txt_type = "appendix"
    else :
        txt_type = "main"
    
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
                    "source": source, 
                    "type": txt_type, 
                    "title": current_title,
                    "subtitle": current_subtitle,
                    "subsection": current_subsection,
                    "subsubsection": current_subsub,
                    "content": it
                })
        buffer.clear()

    for line in lines:
        m = marker_pat.match(line) 
        if m: #If new part detected then run flush buffer
            flush_buffer()
            marker, label = m.group(1), m.group(2).strip()
            #Replace part by new part, affect only downstream parts
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
