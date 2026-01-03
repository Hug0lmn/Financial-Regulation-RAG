import re

def split_numbered_items(block, item_pattern):
    
    # Inside a block of text, split it into subtexts by using item_pattern 
    block_lines = block.strip().split("\n")
    all_subparts = []
    current_subpart = []

    for line in block_lines:
        if item_pattern.match(line): #If a string indicating a new subpart is found then
            if current_subpart : #If there is a previous subpart
                all_subparts.append("".join(current_subpart).strip()) #Add the entire previous subpart
            current_subpart = [line] #Reset current_subpart to new subpart

        else: #If no indication of new subpart
            if current_subpart : #Subpart already composed of txt
                current_subpart.append(line) #Add new line to current subpart
            else: #This condition will only be True once, at the beginning
                current_subpart = [line] 

    if current_subpart : #Add the last subpart of the block, because there is no pattern at the end
        all_subparts.append("".join(current_subpart).strip())

    return "\n\n".join(all_subparts)

def parse(text, name, appendix = False):

    #Item pattern detect a new mini paragraph
    item_pattern = re.compile(r"^([A-Z]?\d+[A-Z]?\.?\d*\.?\d*)\s+") #Lot of optional but need to take into account 13D / 3.2.1 / 2.3 / B.3.1 etc...
    part_pattern = re.compile(r"^(_title_|_subtitle_|_subsection_|_subsubsection_)(.*)$")

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
#            items = split_numbered_items(block,item_pattern)
#            for it in items:
            block = split_numbered_items(block,item_pattern)
            final.append({
                "source": source, 
                "type": txt_type, 
                "title": current_title,
                "subtitle": current_subtitle,
                "subsection": current_subsection,
                "subsubsection": current_subsub,
                "content": block
            })

        buffer.clear()

    for line in lines:
        m = part_pattern.match(line) 
        if m: #If new part detected then run flush buffer
            flush_buffer()
            part, part_name = m.group(1), m.group(2).strip()
            #Replace part by new part, affect only downstream parts
            if part == "_title_": 
                current_title = part_name
                current_subtitle = None
                current_subsection = None
                current_subsub = None
            elif part == "_subtitle_":
                current_subtitle = part_name
                current_subsection = None
                current_subsub = None
            elif part == "_subsection_":
                current_subsection = part_name
                current_subsub = None
            elif part == "_subsubsection_":
                current_subsub = part_name
            continue

        buffer.append(line)

    flush_buffer()
    return final
