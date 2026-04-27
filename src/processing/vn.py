from .base import BaseDocumentProcessor
from bs4 import BeautifulSoup
import re

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
    def extract_text(self, raw_content: str):
        return self._html_to_text(raw_content)
    
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

    def chunk(self, structure: dict) -> list[dict]:
        """Convert structure → list of chunks with metadata"""
        pass

    def process(self, raw_content: str) -> list[dict]:
        """Full Pipeline"""
        pass
    
    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
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
            print("no header")
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

    # df = data.load()
    # import pprint
    # pp = pprint.PrettyPrinter(depth=5, width=120, compact=False)
    # for i in range(3):
    #     html_content = df.iloc[i]["content_html"]
    #     before_clean = processor._html_to_text(html_content)
    #     after_clean = processor._preprocess_text(before_clean)
    #     print("-"*80)
    #     print("Before clean:")
    #     print(before_clean)
    #     print("-"*80)
    #     print("After clean:")
    #     print(after_clean)
    _sample_text = """QUYẾT ĐỊNH
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
    sample_text = """QUY ĐỊNH VỀ QUẢN LÝ VÀ SỬ DỤNG THIẾT BỊ CÔNG NGHỆ TRONG CƠ QUAN NHÀ NƯỚC
Điều 12. Trách nhiệm của cán bộ, công chức trong việc bảo quản thiết bị
Cán bộ, công chức, viên chức được giao sử dụng thiết bị công nghệ (máy tính xách tay, máy tính bảng, điện thoại công vụ) có trách nhiệm bảo quản tài sản công theo đúng quy định của pháp luật về quản lý tài sản nhà nước. Việc sử dụng thiết bị phải đảm bảo đúng mục đích công việc, không được tự ý cho mượn hoặc chuyển giao khi chưa có sự đồng ý của cấp có thẩm quyền.
1. Khi phát hiện thiết bị có dấu hiệu hư hỏng hoặc gặp sự cố kỹ thuật, người sử dụng phải thực hiện các bước sau đây:a) Thông báo ngay cho bộ phận quản trị hệ thống hoặc đơn vị phụ trách công nghệ thông tin của cơ quan để ghi nhận tình trạng sự cố;b) Lập biên bản xác nhận hiện trạng thiết bị, trong đó nêu rõ:
2. Thời điểm phát hiện sự cố;
3. Biểu hiện cụ thể của hỏng hóc (không lên nguồn, lỗi phần mềm, hư hỏng vật lý);
4. Nguyên nhân sơ bộ (nếu xác định được).
c) Phối hợp với bộ phận chuyên môn để tiến hành các thủ tục sửa chữa hoặc thay thế theo quy trình tài chính của đơn vị.
3. Trong trường hợp làm mất mát hoặc hư hỏng thiết bị do lỗi chủ quan, người sử dụng phải chịu trách nhiệm như sau:a) Bồi thường thiệt hại bằng tiền mặt tương đương với giá trị còn lại của thiết bị tại thời điểm xảy ra mất mát, hư hỏng;b) Thực hiện việc sửa chữa và thay thế linh kiện chính hãng nếu thiết bị hư hỏng nhưng vẫn có khả năng phục hồi công năng sử dụng;c) Tùy theo mức độ vi phạm và giá trị tài sản, người vi phạm có thể bị xem xét xử lý kỷ luật theo quy định của Luật Cán bộ, công chức và Luật Viên chức hiện hành.
4. Định kỳ hàng quý, bộ phận quản lý tài sản có trách nhiệm kiểm tra hiện trạng thiết bị và lập báo cáo tổng hợp gửi lãnh đạo cơ quan. Báo cáo phải bao gồm các nội dung:
Tổng số thiết bị đang vận hành tốt;
Danh mục các thiết bị cần bảo trì, bảo dưỡng định kỳ;
Danh sách các thiết bị lỗi thời, cần thực hiện thủ tục thanh lý theo quy định.
Nghiêm cấm các hành vi sau đây đối với việc sử dụng thiết bị công nghệ công vụ:
Tự ý thay đổi cấu hình phần cứng hoặc cài đặt các phần mềm không có bản quyền, phần mềm gây nguy cơ mất an toàn thông tin;
Sử dụng thiết bị để truy cập các trang thông tin điện tử có nội dung độc hại, vi phạm pháp luật hoặc ảnh hưởng đến thuần phong mỹ tục;
Sử dụng dung lượng lưu trữ công vụ vào mục đích lưu trữ dữ liệu cá nhân có kích thước lớn, gây lãng phí tài nguyên hệ thống.
"""
    sample_text = processor._preprocess_text(sample_text)
    segments = processor._split_by_level(sample_text, "điểm")
    for header,content in segments:
        print("-"*80)
        print(f"Header: {header}\n")
        print("-"*80)
        print(f"content: {content}")
    # import pprint
    # pp = pprint.PrettyPrinter(depth=5, width=120, compact=False)
    # structure = processor.parse_structure(sample_text)
    # pp.pprint(structure)
    # print("-"*80)

        
    