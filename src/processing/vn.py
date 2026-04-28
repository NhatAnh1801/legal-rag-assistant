from .base import BaseDocumentProcessor
from bs4 import BeautifulSoup
import re
import time

LEVEL_HIERARCHY = ['phần', 'chương', 'mục', 'tiểu_mục', 'điều','khoản','điểm']
HEADER_PATTERNS = {
    'phần':      re.compile(r'^\s*phần\s+(thứ\s+)?([\divxlcdm]+)', re.UNICODE),
    'chương':    re.compile(r'^\s*chương\s+([\divxlcdm]+)', re.UNICODE),
    'mục':       re.compile(r'^\s*mục\s+(\d+)', re.UNICODE),
    'tiểu_mục':  re.compile(r'^\s*tiểu\s+mục\s+(\d+)', re.UNICODE),
    'điều':      re.compile(r'^\s*điều\s+(\d+)', re.UNICODE),
    'khoản': re.compile(r'^([1-9]\d?(?:\.\d+)?)[\.\-]\s*', re.UNICODE),
    'điểm':      re.compile(r'^([a-zđ])[/\.\)]\s+', re.UNICODE),
}

class VietnameseDocumentProcessor(BaseDocumentProcessor):
    def extract_text(self, raw_content: str) -> str:
        if not raw_content:
            return ""
        if raw_content.strip().startswith("<"):
            return self._html_to_text(raw_content)
        return raw_content  

    def parse_structure(self, text: str) -> dict:
        text = self._preprocess_text(text)
        
        lines = text.splitlines()
        preamble_lines = []
        for line in lines:
            normalized = line.strip().lower()
            found = any(HEADER_PATTERNS[l].match(normalized) for l in LEVEL_HIERARCHY)
            if found:
                break
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines).strip()
        
        return {
            "level": "root",
            "header": "",
            "content": preamble,
            "children": self._build_tree(text, 0)
        }

    def chunk(self, structure: dict, doc_metadata: dict = None) -> list[dict]:
        """Convert structure → list of chunks with metadata"""
        chunks = []
        self._collect_chunks(structure, ancestors=[], chunks=chunks, doc_metadata=doc_metadata or {})
        return chunks

    def process(self, raw_content: str) -> list[dict]:
        """Full ingestion pipeline"""
        start = time.time()
        text = self.extract_text(raw_content)
        end = time.time()
        print(f"Time to extract text: {end - start:.2f}s")
        structure = self.parse_structure(text)
        return self.chunk(structure)
    
    def _collect_chunks(self, node, ancestors, chunks, doc_metadata):
        if not node["children"]:
            breadcrumb = " > ".join(a["header"] for a in ancestors if a["level"] != "root")
            context = (breadcrumb + "\n" if breadcrumb else "") + node["header"]
            chunks.append({
                "content": (context + "\n" + node["content"]).strip(),
                "metadata": {
                    **doc_metadata, "level": node["level"], 
                    "header": node["header"],
                    "parent_header": ancestors[-1]["header"] if len(ancestors) > 1 else ""
                }
            })
        else:
            for child in node["children"]:
                self._collect_chunks(child, ancestors + [node], chunks, doc_metadata)   
        
    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    
    def _preprocess_text(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(':') and cleaned:
                cleaned[-1] = cleaned[-1] + stripped
            else:
                cleaned.append(stripped)
        return "\n".join(cleaned)
    
    def _build_tree(self, text: str, level_idx: int) -> list[dict]:
        if level_idx >= len(LEVEL_HIERARCHY):
            return []
        
        level = LEVEL_HIERARCHY[level_idx]
        segments = self._split_by_level(text, level)
        
        if len(segments) == 1 and segments[0][0] == "":
            return self._build_tree(text, level_idx + 1)
        
        nodes = []
        for header, content in segments:
            node = {
                "level": level,
                "header": header,
                "content": content,
                "children": self._build_tree(content, level_idx + 1)
            }
            nodes.append(node)
        return nodes
    
    def _split_by_level(self, text: str, level: str) -> list[tuple[str, str]]:
        """Split the text by the given level. Return a list of tuples, each tuple contains the level and the text content under that level"""
        
        pattern = HEADER_PATTERNS[level]
        lines = text.splitlines(keepends=True)
        
        # Find the indices of headers
        header_indices = []
        for i, line in enumerate(lines):
            if pattern.match(line.strip().lower()):
                header_indices.append(i)

        # No header -> there is only 1 segment
        if not header_indices:
            return [("", text)]
        
        segments = []
        
        for i, idx in enumerate(header_indices):
            full_header_line = lines[idx].strip()
            end = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            body = "".join(lines[idx + 1:end]).strip()

            if level in {"khoản", "điểm"}:
                parts = full_header_line.split(maxsplit=1)
                header_token = parts[0].rstrip(".-")
                inline_content = parts[1].strip() if len(parts) > 1 else ""
            else:
                m = HEADER_PATTERNS[level].match(full_header_line.lower())
                header_token = m.group(0).strip()
                inline_content = full_header_line[len(m.group(0)):].lstrip(':. ').strip()
            content = (inline_content + "\n" + body).strip() if inline_content else body
            segments.append((header_token, content))
        
        return segments

    
from src.data_loader.vn import VietnameseDataLoader
if __name__ == "__main__":
    data = VietnameseDataLoader()
    processor = VietnameseDocumentProcessor()

    sample_text = """QUYẾT ĐỊNH
Về việc chuyển giao nhiệm vụ quản lý Nhà nước về đất lâm nghiệp
từ Sở Nông nghiệp và Phát triển Nông thôn và Chi cục kiểm lâm tỉnh sang Sở Địa chính để quản lý
ỦY BAN NHÂN DÂN TỈNH LÂM ĐỒNG
Căn cứ Luật Tổ chức HĐND và UBND (sửa đổi) ngày 21/06/1994;
Căn cứ Lệnh số: 58 LCT/HĐNN ngày 19/8/1991 của Chủ tịch nước, "về việc công bố Luật Bảo vệ và phát triển rừng";
Căn cứ Quyết định số: 245/QĐ-TTg ngày 21/12/1998 của Thủ tướng Chính phủ, về thực hiện trách nhiệm quản lý Nhà nước của các cấp về rừng và đất lâm nghiệp;
Theo đề nghị của Ban TCCQ tỉnh Lâm Đồng, sau khi thống nhất với Sở NN&PTNT, Chi cục Kiểm lâm và Sở Địa chính tỉnh Lâm Đồng,
QUYẾT ĐỊNH:
Điều 1
: Chuyển giao nhiệm vụ quản lý Nhà nước về đất lâm nghiệp từ Sở Nông nghiệp & PTNT và Chi cục kiểm lâm sang Sở Địa chính để quản lý.
Điều 2
: Sở Địa chính có nhiệm vụ giúp UBND tỉnh Lâm Đồng thực hiện trách nhiệm quản lý nhà nước về đất lâm nghiệp trên địa bàn tỉnh Lâm Đồng cụ thể như sau:
2.1- Tổ chức việc điều tra, lập bản đồ phân định ranh giới về đất lâm nghiệp trên địa bàn tỉnh theo quy định của chính phủ và hướng dẫn của Tổng cục Địa chính.
Phối hợp với Sở Nông nghiệp và Phát triển Nông thôn Chỉ đạo, hướng dẫn UBND các huyện, TX Bảo Lộc, TP Đà Lạt theo dõi, lập báo cáo thống kê biến động về đất lâm nghiệp và tổng hợp báo cáo UBND tỉnh.
2.2- Cùng Sở Nông nghiệp và Phát triển Nông thôn lập quy hoạch và kế họach sử dụng đất lâm nghiệp của tỉnh trình UBND tỉnh, Hội đồng nhân dân tỉnh thông qua trước khi trình Chính phủ xét duyệt
.
2.3- Hướng dẫn UBND các huyện, TX. Bảo Lộc, TP. Đà Lạt và các đơn vị có liên quan lập quy hoạch, kế hoạch sử dụng đất lâm nghiệp và thẩm định trình UBND tỉnh phê duyệt.
2.4- Tham mưu giúp UBND tỉnh thực hiện giao đất lâm nghiệp, thu hồi đất lâm nghiệp, cấp giấy chứng nhận quyền sử dụng đất và các nghiệp vụ khác liên quan đến việc quản lý đất lâm nghiệp giao cho các thành phần kinh tế khác theo quy định của pháp luật; hưởng dẫn
Ủ
y ban nhân dân cấp huyện thực hiện giao đất, cho thuê đất lâm nghiệp đối với các hộ gia đình, cá nhân theo đúng chính sách, chế độ quy định của nhà nước.
2.5- Kiểm tra, thanh tra và xử lý các vi phạm trong việc chấp hành pháp luật, chính sách về quản lý sử dụng đất lâm nghiệp; giải quyết các tranh chấp về đất lâm nghiệp theo đúng thẩm quyền và quy định của luật.
Điều 3
: Giao các ông Giám đốc Sở NN&PTNT; Chi cục trưởng chi cục kiểm lâm, thủ trưởng các ngành có liên quan của tỉnh triển khai thực hiện công tác chuyển giao nhiệm vụ quản lý nhà nước về đất lâm nghiệp từ đơn vị mình cho Giám đốc Sở Địa chính (bao gồm các hồ sơ, tài liệu, bản đồ có liên quan đến công tác quản lý đất lâm nghiệp; hồ sơ giao đất lâm nghiệp cho các thành phần kinh tế và tổ chức nhà nước trong thời gian qua; hồ sơ, Bản đồ phân định đất Nông, đất lâm của tỉnh); để Sở Địa chính thực hiện nhiệm vụ giúp UBND tỉnh thực hiện trách nhiệm quản lý nhà nước về đất lâm nghiệp trên địa bàn tỉnh Lâm Đồng theo nội dung Quyết định số: 245/QĐ-TTg ngày 21/12/1998 của Thủ tướng Chính phủ, "về thực hiện trách nhiệm quản lý Nhà nước của các cấp về đất lâm nghiệp"; Công tác
chuyển giao tiếp nhận phải hoàn thành trước ngày 31/7/1999;
Điều 4
: Quyết định này có hiệu lực kể từ ngày ký. Mọi quyết định trước đây trái với Quyết định này đều hết hiệu lực thi hành.
Điều 5
: Các ông: Chánh VP UBND tỉnh, Trưởng Ban TCCQ tỉnh, Giám đốc các Sở: Tài chính vật giá; NN&PTNT; Địa chính; Chi cục trưởng chi cục kiểm lâm cùng thủ trưởng các ngành chức năng có liên quan của tỉnh, Chủ tịch UBND các huyện, thị xã Bảo Lộc và thành phố Đà Lạt căn cứ Quyết định thi hành./.
"""
    df = data.load(batch_size=10, offset=0)
    html = df.iloc[0]["content_html"]

    start_1 = time.time()
    processor._html_to_text(html)
    end_1 = time.time()
    print(f"_html_to_text: {end_1 - start_1:.6f}s/doc")

    start_2 = time.time()
    text = processor._html_to_text(html)

    structure = processor.parse_structure(text)
    end_2 = time.time()
    print(f"parse_structure: {end_2 - start_2:.6f}s/doc")
    
    start_3  = time.time()
    processor.chunk(structure)
    end_3 = time.time() 
    print(f"chunk: {end_3 - start_3:.6f}s/doc")
    print(f"Total time: {end_3 - start_1:.6f}s/doc")
    
    # total_df = len(data.load())
    # print(f"total df nums: {total_df}")
    
    # start = time.time()
    # df = data.load(batch_size=10, offset=0)
    # end = time.time()
    # print(f"Time to load data: {end - start:.2f}s")
    
    # num_samples = len(df)
    # start = time.time()
    # for i in range(num_samples):
    #     print(f"\n======= Sample {i} | {df.iloc[i]['loai_van_ban']} =======")
    #     chunks = processor.process(df.iloc[i]["content_html"])
    #     for chunk in chunks:
    #         print("-" * 80)
    #         print(chunk)
    # end = time.time()
    # print(f"\nTotal time: {end - start:.2f}s for {num_samples} docs")

 

    

        
    