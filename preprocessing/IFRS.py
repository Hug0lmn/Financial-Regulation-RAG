import pdfplumber
from pathlib import Path
import re
import numpy as np
import fitz

#The first page that needs ocr will be deleted after and only the 73 really needs ocr. I actually used ChatGPT to get directly the text from the page directly and the layout as it needed really specific instructions to get good results while some texts were missing whatever I'm doing
ifrs_9_seventy_four = '''IFRS 9\n
_title_Derecognition of financial assets (Section 3.2)\n
\n
B3.2.1\n
The following flow chart illustrates the evaluation of whether and to what extent a financial asset is derecognised.\n
\n
Consolidate all subsidiaries [Paragraph 3.2.1]\n
\n
Determine whether the derecognition principles below are applied to a part or all of an asset (or group of similar assets) [Paragraph 3.2.2]\n
\n
Have the rights to the cash flows from the asset expired? [Paragraph 3.2.3(a)]\n
Yes → Derecognise the asset\n
No → Continue\n
\n
Has the entity transferred its rights to receive the cash flows from the asset? [Paragraph 3.2.4(a)]\n
Yes → Continue\n
No → Continue\n
\n
Has the entity assumed an obligation to pay the cash flows from the asset that meets the conditions in paragraph 3.2.5? [Paragraph 3.2.4(b)]\n
No → Continue to recognise the asset\n
Yes → Continue\n
\n
Has the entity transferred substantially all risks and rewards? [Paragraph 3.2.6(a)]\n
Yes → Derecognise the asset\n
No → Continue\n
\n
Has the entity retained substantially all risks and rewards? [Paragraph 3.2.6(b)]\n
Yes → Continue to recognise the asset\n
No → Continue\n
\n
Has the entity retained control of the asset? [Paragraph 3.2.6(c)]\n
No → Derecognise the asset\n
Yes → Continue to recognise the asset to the extent of the entity’s continuing involvement\n
\n
A436\n
© IFRS Foundation\n
'''

def to_markdown(headers, cols, continued = False):
    md = ""

    if not continued :
        for i in headers :
            md += f"| {i} "
        md+="|\n"

        for i in headers :
            md += "|---"
        md+="|\n"

    for i in cols :
        i = re.sub(r"([a-z])\n([a-z])", r"\1 \2", i)
        i = re.sub(r"-\n([a-z])", r"\1", i)
        i = i.replace("\n", "<br>")

        md += f"| {i} "

    return md

def transform_table_to_use(pdf, num, text_blocks, tables) :

#    for num in tables_ :
    
    table = pdf.pages[num].find_table().cells 
    
    y_min = 10000
    y_max = 0
    for elem in table :
        y_min = min(elem[1],y_min)
        y_max = max(elem[3],y_max)

    before =[]
    after = []
        
    for i in pdf.pages[num].extract_words() :
        if i["top"] < y_min and len(before) < 10:
            before.append(i["text"])
        if i["bottom"] > y_max and len(after) < 10 :
            after.append(i["text"])

        #Reconstruct the true page
    new_page = ""
    tablemm = tables[num][0]
    headers = tablemm[0]
    cols = tablemm[1]
        
    if len(after) > 5 :
        past_table = False

        after_text = " ".join(after[:5])
        text = text_blocks[num]

        for j in text.split("\n") :
            if past_table :
                new_page += j

            elif after_text in j :
                past_table = True
                new_page += after_text+j.split(after_text)[1]
        
    else : 
        if "...continued" in before :
            new_page += to_markdown(headers, cols, continued=True)
        else :
            new_page += to_markdown(headers,cols)

        text_blocks[num] = new_page

    return new_page

def appendix_def (page) :
    text = []
    ll = page.get_text_blocks()
    target = float("inf")

    for j in range(len(ll)) :
        current_text = ll[j]
        if j == target :
            continue

        elif re.findall(r"(^[a-z,A-Z].*\n(?=[A-Z]))",current_text[4]) : #Def where the whole def word is in the text
            test = re.sub(r"([a-z,A-Z])?\n(?=[A-Z])",r"\1 : ",current_text[4])
            test = re.sub(r"([a-z,A-Z,\)])\n(?=[a-z,\(])",r"\1 ",test)
            test = test.strip()
            text.append(test)
    
        elif re.findall(r"(^[a-z].*\n\b[a-z]+)",current_text[4]) : #Def where only the word is in the 
            target = j+1
            next_text = ll[j+1][4]
            new_text = current_text[4].replace("\n"," ")+": "+next_text
        
            text.append(new_text)
    
        else : 
            text.append(re.sub(r"\n"," ",current_text[4]))

    text = "\n".join(text)
    
    return text

def extract_text_from_ifrs_lines(page) :
    
    global_ = []

    for j in range(1,len(page)) :

        word = page[j]
        word_1 = page[j-1]

        word["height"] = round(word["height"],1)
        word_1["height"] = round(word_1["height"],1)

            #do not count footpage/headers 
        if (word["top"] < 105) or (word["top"] > 710) or (word["height"] >12):
            continue

        if global_ == [] : 
            if word["height"] >= 11 :
                global_.extend(f'_title_{word["text"]}')
            elif word["height"] > 9 :
                global_.extend(f'_subtitle_{word["text"]}')
            else : 
                global_.extend(word["text"])

        elif (word["top"] == word_1["top"]) : #Same line same part
            global_.extend(word["text"])

            # Between inf then same part
        elif abs(word_1["bottom"] - word["bottom"]) < 13 : 
            global_.extend(word["text"])     

        elif abs(word_1["bottom"] - word["bottom"]) > 13 : 

            if (word["height"] >= 11) :
                global_.extend(f'\n_title_{word["text"]}')
            elif word["height"] > 9 :
                global_.extend(f'\n_subtitle_{word["text"]}')
            elif (word["height"] < 8) :
                global_.extend(f'\n_footnote_{word["text"]}')
                
            else :
                global_.extend(f'\n{word["text"]}')

        global_.extend(" ")
            
    global_ = "".join(global_)

    return global_

def global_process_ifrs(file_path) :

    #First assume every page contains only structured texts with one column 
    tables = []
    text_blocks = []
    words = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text_blocks.append(page.extract_text())
            tables.append(page.extract_tables()) #Will help identify page with table to perform specific cleaning
            words.append(page.extract_words())

    doc = fitz.open(file_path)

    #To know when to use OCR
    ocr_needed = []
    tables_ = []

    for page in range(len(doc)) :
        if len(doc[page].get_drawings()) > 5 : 

            if not tables[page] : #Tables not recognized by pdfplumber
                ocr_needed.append(page)
            else :
                tables_.append(page)
    
    
    text_final = []
    beginning = False
    definitions_ = False

    for j in range(len(text_blocks)) :

        if "IFRS_9" in str(file_path) :
            if j == 73 :
                text_final.append(ifrs_9_seventy_four)
                continue

        if j in tables_ :
            text_final.append(transform_table_to_use(pdf, j, text_blocks, tables))
            continue

        if re.findall(r"Objective\n",text_blocks[j]) :
            beginning = True
        elif re.findall(r"Appendix A\nDefined terms",text_blocks[j]) :
            definitions_ = True
        elif re.findall(r"Appendix B\nApplication guidance",text_blocks[j]) :
            definitions_ = False
        elif re.findall(r"Appendix [A-Z]\nAmendments ",text_blocks[j]) :
            beginning = False
    

        if beginning : 
            if definitions_ :
                text_final.append(appendix_def(doc[j]).strip())
            else :
                text_final.append(extract_text_from_ifrs_lines(words[j]).strip())

    return text_final