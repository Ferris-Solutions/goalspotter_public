import io
import re
import bs4
import numpy
import pandas
import requests
import pdfminer.pdfparser
import pdfminer.pdfinterp
import pdfminer.high_level
import pdfminer.pdfdocument


class Document:
    
    def __init__(self, url, content_type=None, annotations=None):
        self.url = url
        self.content_type = content_type
        self.annotations = annotations
    
    def read_local_file(self):
        return open(self.url, "rb").read()
    
    def request_url(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.3"
        }
        response = requests.get(self.url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        content = None
        if response.status_code == 200:
            content_type = response.headers.get("content-type")
            if "html" in content_type:
                self.content_type = "html" 
            elif "pdf" in content_type:
                self.content_type = "pdf"
            else:
                print(f"Unsupported content type for {self.url}: {content_type}")
            content = response.content
        return content
    
    def parse_content(self, content):
        
        def parse_html(html):
            parsed_html = bs4.BeautifulSoup(html, "html.parser")
            removed_tags=["style", "script"]
            for tag in parsed_html(removed_tags):
                tag.decompose()
            text = parsed_html.body.get_text().strip()
            return text

        def parse_pdf(pdf):
            f = io.BytesIO(pdf)
            parser = pdfminer.pdfparser.PDFParser(f)
            document = pdfminer.pdfdocument.PDFDocument(parser)
            page_count = pdfminer.pdfinterp.resolve1(document.catalog["Pages"])["Count"] 
            pages = []
            for page_number in range(page_count): 
                text = pdfminer.high_level.extract_text(f, page_numbers=[page_number])
                pages.append(text)
            return pages

        if self.content_type == "html":
            return [parse_html(content)]
        if self.content_type == "pdf":
            return parse_pdf(content)
        
    def segment_text(self, parsed_content):
        text_blocks = []
        for text in parsed_content:
            newlines = re.findall("[\n]+", text)
            newlines_count = [n.count("\n") for n in newlines]
            try:
                if self.content_type == "html":
                    threshold = round(numpy.median(newlines_count))
                if self.content_type == "pdf":
                    threshold = max(newlines_count)
            except:
                threshold = 1
            temp_text_blocks = re.split("[\n]{" + str(threshold) + ",}", text)
            temp_text_blocks = [tb.strip() for tb in temp_text_blocks if tb.strip()]
            text_blocks.extend(temp_text_blocks)
        return text_blocks
    
    def label_text_blocks(self, text_blocks):
        
        def text_preprocessor(s):
            s = s.lower()
            s = s.strip()
            s = re.sub("[^a-zA-Z]+", "", s)
            return s
        
        rdf = pandas.DataFrame({"URL": self.url, "Content Type": self.content_type, "Text Blocks": text_blocks})
        rdf["Text Blocks"] = rdf["Text Blocks"].astype(str)
        rdf["Preprpcessed Text Blocks"] = rdf["Text Blocks"].apply(text_preprocessor)
        preprocessed_goals = [text_preprocessor(goal) for goal in self.annotations]
        pattern = "|".join(preprocessed_goals)
        rdf["Goal"] = rdf["Preprpcessed Text Blocks"].str.contains(pattern, case=False)
        rdf["Goal"] = rdf["Goal"].astype(int)
        rdf.drop("Preprpcessed Text Blocks", axis=1, inplace=True)
        return rdf
