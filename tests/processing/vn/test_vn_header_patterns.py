from src.processing.vn import VietnameseDocumentProcessor
import pytest
DIEM_CASES = [
    # Happy path cases
    pytest.param("a) Nội dung điểm a", {"header_name": "điểm", "header_value": "a", "remainder": "Nội dung điểm a"}, id="diem_a_paren"),
    pytest.param("b) Giải thích điểm b", {"header_name": "điểm", "header_value": "b", "remainder": "Giải thích điểm b"}, id="diem_b_paren"),
    pytest.param("c. Điểm c dùng dấu chấm", {"header_name": "điểm", "header_value": "c", "remainder": "Điểm c dùng dấu chấm"}, id="diem_c_dot"),
    pytest.param("đ) Tiếng Việt có chữ đ", {"header_name": "điểm", "header_value": "đ", "remainder": "Tiếng Việt có chữ đ"}, id="diem_d_vietnamese"),
    pytest.param("g- Nội dung với gạch ngang", {"header_name": "điểm", "header_value": "g", "remainder": "Nội dung với gạch ngang"}, id="diem_g_dash"),
    pytest.param("  a) có khoảng trắng đầu dòng", {"header_name": "điểm", "header_value": "a", "remainder": "có khoảng trắng đầu dòng"}, id="diem_leading_whitespace"),
    pytest.param("aa) hai ký tự trong label", {"header_name": "điểm", "header_value": "aa", "remainder": "hai ký tự trong label"}, id="diem_multi_letter_label"),
    pytest.param("i. chữ i thường không phải La Mã", {"header_name": "điểm", "header_value": "i", "remainder": "chữ i thường không phải La Mã"}, id="diem_lowercase_i_not_roman"),
    # Tiểu mục (IN HOA)
    pytest.param("A. MỤC LỚN", {"header_name": "tiểu_mục", "header_value": "A", "remainder": "MỤC LỚN"}, id="resolve_uppercase_A_dot"),
    pytest.param("B) Tiểu mục chữ B", {"header_name": "tiểu_mục", "header_value": "B", "remainder": "Tiểu mục chữ B"}, id="resolve_uppercase_B_paren"),
    pytest.param("Đ) Chữ Đ hoa", {"header_name": "tiểu_mục", "header_value": "Đ", "remainder": "Chữ Đ hoa"}, id="resolve_uppercase_D_stroke"),
    # Mục (La Mã)
    pytest.param("I. Mục La Mã một", {"header_name": "mục", "header_value": "I", "remainder": "Mục La Mã một"}, id="resolve_roman_I"),
    pytest.param("II. Mục La Mã hai", {"header_name": "mục", "header_value": "II", "remainder": "Mục La Mã hai"}, id="resolve_roman_II"),
    pytest.param("IV. Mục La Mã bốn", {"header_name": "mục", "header_value": "IV", "remainder": "Mục La Mã bốn"}, id="resolve_roman_IV"),
    pytest.param("IX. Mục La Mã chín", {"header_name": "mục", "header_value": "IX", "remainder": "Mục La Mã chín"}, id="resolve_roman_IX"),
    pytest.param("X. Mục La Mã mười", {"header_name": "mục", "header_value": "X", "remainder": "Mục La Mã mười"}, id="resolve_roman_X"),
    # False postive cases
    pytest.param("Không phải header điểm", None, id="no_match_plain_text"),
    pytest.param("a)thiếu khoảng trắng sau delimiter", None, id="no_match_missing_space_after_delimiter"),
    pytest.param("- Biểu dương trong đơn vị", None, id="no_match_bullet_dash_prefix"),
    pytest.param("abc thiếu delimiter", None, id="no_match_no_delimiter"),
    pytest.param("1.000 đồng trong bảng giá", None, id="no_match_number_with_thousand_separator"),
    pytest.param("v.v thiếu delimiter", None, id="no_match_no_delimiter_after_v"),
    pytest.param("TP.HCM đây là thành phố của Việt Nam", None, id="no_match_city_name_with_dot"),
    pytest.param("1-100 đây là số từ 1 đến 100", None, id="no_match_number_with_dash"),
]

KHOAN_CASES = [
    # Happy path cases
    pytest.param("1. Đây là khoản 1", {"header_name": "khoản", "header_value": "1", "remainder": "Đây là khoản 1"}, id="khoan_1_dot"),
    pytest.param("2- Nội dung khoản 2", {"header_name": "khoản", "header_value": "2", "remainder": "Nội dung khoản 2"}, id="khoan_2_dash"),
    pytest.param("3/ Nội dung khoản 3 dùng slash", {"header_name": "khoản", "header_value": "3", "remainder": "Nội dung khoản 3 dùng slash"}, id="khoan_3_slash"),
    pytest.param("4) Khoản 4 dùng dấu ngoặc đơn", {"header_name": "khoản", "header_value": "4", "remainder": "Khoản 4 dùng dấu ngoặc đơn"}, id="khoan_4_paren"),
    pytest.param("10. Khoản số lớn hơn 9", {"header_name": "khoản", "header_value": "10", "remainder": "Khoản số lớn hơn 9"}, id="khoan_10_dot"),
    pytest.param("5a. Khoản với hậu tố chữ cái", {"header_name": "khoản", "header_value": "5a", "remainder": "Khoản với hậu tố chữ cái"}, id="khoan_5a_alpha"),
    pytest.param("3. Ở trong những abc", {"header_name": "khoản", "header_value": "3", "remainder": "Ở trong những abc"}, id="no_match_number_with_abc_khoan"),
    pytest.param("    6)   Khoản với khoảng trắng đầu dòng", {"header_name": "khoản", "header_value": "6", "remainder": "Khoản với khoảng trắng đầu dòng"}, id="khoan_leading_whitespace"),
    pytest.param("8.1. Khoản với số thập phân dot", {"header_name": "khoản", "header_value": "8.1", "remainder": "Khoản với số thập phân dot"}, id="khoan_decimal_dot"),
    pytest.param("9a) Khoản với hậu tố chữ và dấu ngoặc", {"header_name": "khoản", "header_value": "9a", "remainder": "Khoản với hậu tố chữ và dấu ngoặc"}, id="khoan_kytu_dot_paren"),
    pytest.param("1- Người ngoại kiều", {"header_name": "khoản", "header_value": "1", "remainder": "Người ngoại kiều"}, id="khoan_legacy_1_dash"),
    pytest.param("2đ. Nội dung", {"header_name": "khoản", "header_value": "2đ", "remainder": "Nội dung"}, id="khoan_vn_suffix"), 
    pytest.param("11) Nội dung", {"header_name": "khoản", "header_value": "11", "remainder": "Nội dung"}, id="khoan_11_paren"),
    pytest.param("12312312/ số lớn", {"header_name": "khoản", "header_value": "12312312", "remainder": "số lớn"}, id="khoan_large_number_slash"),
    pytest.param("1. Ở những cái...", {"header_name": "khoản", "header_value": "1", "remainder": "Ở những cái..."}, id="khoan_1_dot_with_vietnamese_text"),
    # False positive cases
    pytest.param("01. Bắt đầu bằng số 0 không hợp lệ", None, id="khoan_leading_zero"),
    pytest.param("100 đồng là số tiền", None, id="khoan_money_not_khoan"),
    pytest.param("1.000 Đây là số nghìn", None, id="khoan_thousands_separator"),
    pytest.param("- Khoản với dấu gạch đầu", None, id="khoan_dash_instead_of_number"),
    pytest.param("7-1. Khoản với số thập phân", None, id="khoan_decimal_dash"),
    pytest.param("1, Định dạng sai với dấu phẩy", None, id="khoan_wrong_delimiter_comma"),
    pytest.param("1.2 Nội dung", None, id="khoan_subclause_decimal"),
    pytest.param("1.", None, id="khoan_no_text_after_dot"),           
    pytest.param("1.  ", None, id="khoan_only_spaces"),       
    pytest.param("1.2 Nội dung", None, id="khoan_subclause_decimal"),
    pytest.param("21/6/1994 Căn cứ", None, id="khoan_date_slash")
]

DIEU_CASES = [
    # Happy path cases
    pytest.param("Điều 1. Nội dung điều 1", {"header_name": "điều", "header_value": "1", "remainder": "Nội dung điều 1"}, id="dieu_1_dot"),
    pytest.param("Điều 2: Quy định chung", {"header_name": "điều", "header_value": "2", "remainder": "Quy định chung"}, id="dieu_2_colon"),
    pytest.param("Điều 10. Số hai chữ số", {"header_name": "điều", "header_value": "10", "remainder": "Số hai chữ số"}, id="dieu_10_dot"),
    pytest.param("Điều 5a. Hậu tố chữ cái", {"header_name": "điều", "header_value": "5a", "remainder": "Hậu tố chữ cái"}, id="dieu_5a_suffix"),
    pytest.param("Điều 12đ. Hậu tố đ", {"header_name": "điều", "header_value": "12đ", "remainder": "Hậu tố đ"}, id="dieu_12d_vietnamese_suffix"),
    pytest.param("  Điều 3. Có khoảng trắng đầu dòng", {"header_name": "điều", "header_value": "3", "remainder": "Có khoảng trắng đầu dòng"}, id="dieu_leading_whitespace"),
    pytest.param("ĐIỀU 7. Viết hoa toàn bộ", {"header_name": "điều", "header_value": "7", "remainder": "Viết hoa toàn bộ"}, id="dieu_uppercase_keyword"),
    pytest.param("điều 8. Chữ thường", {"header_name": "điều", "header_value": "8", "remainder": "Chữ thường"}, id="dieu_lowercase_keyword"),
    pytest.param("Điều 1.", {"header_name": "điều", "header_value": "1", "remainder": ""}, id="dieu_1_dot_only"),
    pytest.param("Điều 2:", {"header_name": "điều", "header_value": "2", "remainder": ""}, id="dieu_2_colon_only"),
    pytest.param("Điều 1", {"header_name": "điều", "header_value": "1", "remainder": ""}, id="dieu_1_no_delimiter"),
    pytest.param("Điều 100. Điều số lớn", {"header_name": "điều", "header_value": "100", "remainder": "Điều số lớn"}, id="dieu_100_dot"),
    pytest.param("Điều 4- Phạm vi áp dụng", {"header_name": "điều", "header_value": "4", "remainder": "Phạm vi áp dụng"}, id="dieu_4_dash_delimiter"),
    # False positive cases
    pytest.param("Điều 1 của Điều lệ nói: nội dung", None, id="dieu_reference_cua_dieu_le"),
    pytest.param("Điều 2 quy định rõ việc miễn", None, id="dieu_reference_quy_dinh"),
    pytest.param('Điều 9 quy định: "Trích dẫn trong câu"', None, id="dieu_reference_quy_dinh_colon_in_prose"),
    pytest.param("Căn cứ Điều 10 Luật tổ chức", None, id="dieu_mid_sentence_can_cu"),
    pytest.param("Theo điều 5 của luật liên quan", None, id="dieu_mid_sentence_theo"),
    pytest.param("Điều", None, id="dieu_missing_number"),
    pytest.param("Điều abc không phải số", None, id="dieu_non_numeric_value"),
    pytest.param("Điềm 1. Lỗi chính tả", None, id="dieu_typo_diem"),
    pytest.param("Không phải header điều", None, id="dieu_plain_text"),
]

TIEU_MUC_CASES = [
    # Happy path — "Tiểu mục" + số
    pytest.param("Tiểu mục 1. Nội dung", {"header_name": "tiểu_mục", "header_value": "1", "remainder": "Nội dung"}, id="tieu_muc_1_dot"),
    pytest.param("Tiểu mục 2: Tiêu đề", {"header_name": "tiểu_mục", "header_value": "2", "remainder": "Tiêu đề"}, id="tieu_muc_2_colon"),
    pytest.param("Tiểu mục 10. Hai chữ số", {"header_name": "tiểu_mục", "header_value": "10", "remainder": "Hai chữ số"}, id="tieu_muc_10_dot"),
    pytest.param("Tiểu mục 3a. Hậu tố", {"header_name": "tiểu_mục", "header_value": "3a", "remainder": "Hậu tố"}, id="tieu_muc_3a_suffix"),
    pytest.param("TIỂU MỤC 4. Viết hoa", {"header_name": "tiểu_mục", "header_value": "4", "remainder": "Viết hoa"}, id="tieu_muc_uppercase_keyword"),
    pytest.param("tiểu mục 5. Chữ thường", {"header_name": "tiểu_mục", "header_value": "5", "remainder": "Chữ thường"}, id="tieu_muc_lowercase_keyword"),
    pytest.param("  Tiểu mục 6. Có indent", {"header_name": "tiểu_mục", "header_value": "6", "remainder": "Có indent"}, id="tieu_muc_leading_whitespace"),
    pytest.param("Tiểu mục 1.", {"header_name": "tiểu_mục", "header_value": "1", "remainder": ""}, id="tieu_muc_1_dot_only"),
    pytest.param("Tiểu mục 2:", {"header_name": "tiểu_mục", "header_value": "2", "remainder": ""}, id="tieu_muc_2_colon_only"),
    pytest.param("Tiểu mục 7", {"header_name": "tiểu_mục", "header_value": "7", "remainder": ""}, id="tieu_muc_7_no_delimiter"),
    pytest.param("Tiểu mục 8- Phạm vi", {"header_name": "tiểu_mục", "header_value": "8", "remainder": "Phạm vi"}, id="tieu_muc_8_dash"),
    pytest.param("Tiểu mục 12đ. Hậu tố đ", {"header_name": "tiểu_mục", "header_value": "12đ", "remainder": "Hậu tố đ"}, id="tieu_muc_vn_suffix"),
    # False positive
    pytest.param("Tiểu mục", None, id="tieu_muc_missing_number"),
    pytest.param("Tiểumục 1. Không có khoảng", None, id="tieu_muc_typo_no_space"),
    pytest.param("Tiểu mục abc không phải số", None, id="tieu_muc_non_numeric"),
    pytest.param("Tiểu mục 1 của mục khác", None, id="tieu_muc_reference_cua"),
    pytest.param("Tiểu mục 2 quy định thêm", None, id="tieu_muc_reference_quy_dinh"),
    pytest.param("Căn cứ tiểu mục 1", None, id="tieu_muc_mid_sentence"),
    pytest.param("Tiểu mục I. La Mã không hỗ trợ", None, id="tieu_muc_roman_not_supported"),
    pytest.param("A. MỤC LỚN", {"header_name": "tiểu_mục", "header_value": "A", "remainder": "MỤC LỚN"}, id="tieu_muc_via_diem_pattern_not_keyword"),
    pytest.param("Không phải tiểu mục", None, id="tieu_muc_plain_text"),
]

MUC_CASES = [
    # Happy path — "Mục" + số hoặc La Mã
    pytest.param("Mục 1. Nội dung mục 1", {"header_name": "mục", "header_value": "1", "remainder": "Nội dung mục 1"}, id="muc_1_dot"),
    pytest.param("Mục 2: Quy định chung", {"header_name": "mục", "header_value": "2", "remainder": "Quy định chung"}, id="muc_2_colon"),
    pytest.param("Mục 10. Số hai chữ số", {"header_name": "mục", "header_value": "10", "remainder": "Số hai chữ số"}, id="muc_10_dot"),
    pytest.param("Mục 3a. Hậu tố chữ cái", {"header_name": "mục", "header_value": "3a", "remainder": "Hậu tố chữ cái"}, id="muc_3a_suffix"),
    pytest.param("Mục I. La Mã một", {"header_name": "mục", "header_value": "I", "remainder": "La Mã một"}, id="muc_roman_I"),
    pytest.param("Mục II: La Mã hai", {"header_name": "mục", "header_value": "II", "remainder": "La Mã hai"}, id="muc_roman_II"),
    pytest.param("Mục IV. La Mã bốn", {"header_name": "mục", "header_value": "IV", "remainder": "La Mã bốn"}, id="muc_roman_IV"),
    pytest.param("Mục IX. La Mã chín", {"header_name": "mục", "header_value": "IX", "remainder": "La Mã chín"}, id="muc_roman_IX"),
    pytest.param("MỤC 5. Viết hoa", {"header_name": "mục", "header_value": "5", "remainder": "Viết hoa"}, id="muc_uppercase_keyword"),
    pytest.param("mục 6. Chữ thường", {"header_name": "mục", "header_value": "6", "remainder": "Chữ thường"}, id="muc_lowercase_keyword"),
    pytest.param("  Mục 7. Có indent", {"header_name": "mục", "header_value": "7", "remainder": "Có indent"}, id="muc_leading_whitespace"),
    pytest.param("Mục 1.", {"header_name": "mục", "header_value": "1", "remainder": ""}, id="muc_1_dot_only"),
    pytest.param("Mục 2:", {"header_name": "mục", "header_value": "2", "remainder": ""}, id="muc_2_colon_only"),
    pytest.param("Mục 8- Phạm vi", {"header_name": "mục", "header_value": "8", "remainder": "Phạm vi"}, id="muc_8_dash_delimiter"),
    pytest.param("Mục IX: Ở những...", {"header_name": "mục", "header_value": "IX", "remainder": "Ở những..."}, id="muc_roman_IX_colon"),
    # False positive
    pytest.param("Mục", None, id="muc_missing_index"),
    pytest.param("Mục abc không phải số", None, id="muc_non_numeric_index"),
    pytest.param("Mục 1 của chương trước", None, id="muc_reference_cua"),
    pytest.param("Mục 2 quy định chi tiết", None, id="muc_reference_quy_dinh"),
    pytest.param("Theo mục 1 của luật", None, id="muc_mid_sentence"),
    pytest.param("Mục đích và phạm vi", None, id="muc_prose_muc_dich"),
    pytest.param("I. Không có từ Mục", {"header_name": "mục", "header_value": "I", "remainder": "Không có từ Mục"}, id="muc_roman_via_diem_not_keyword"),
    pytest.param("Không phải header mục", None, id="muc_plain_text"),
]

CHUONG_CASES = [
    # Happy path
    pytest.param("Chương I. Quy định chung", {"header_name": "chương", "header_value": "I", "remainder": "Quy định chung"}, id="chuong_roman_I_dot"),
    pytest.param("Chương II: Nghĩa vụ", {"header_name": "chương", "header_value": "II", "remainder": "Nghĩa vụ"}, id="chuong_roman_II_colon"),
    pytest.param("Chương IV. La Mã bốn", {"header_name": "chương", "header_value": "IV", "remainder": "La Mã bốn"}, id="chuong_roman_IV"),
    pytest.param("Chương 1. Nội dung chương 1", {"header_name": "chương", "header_value": "1", "remainder": "Nội dung chương 1"}, id="chuong_1_dot"),
    pytest.param("Chương 10. Hai chữ số", {"header_name": "chương", "header_value": "10", "remainder": "Hai chữ số"}, id="chuong_10_dot"),
    pytest.param("Chương 2a. Hậu tố chữ", {"header_name": "chương", "header_value": "2a", "remainder": "Hậu tố chữ"}, id="chuong_2a_suffix"),
    pytest.param("CHƯƠNG 3. Viết hoa", {"header_name": "chương", "header_value": "3", "remainder": "Viết hoa"}, id="chuong_uppercase_keyword"),
    pytest.param("chương 4. Chữ thường", {"header_name": "chương", "header_value": "4", "remainder": "Chữ thường"}, id="chuong_lowercase_keyword"),
    pytest.param("  Chương 5. Có indent", {"header_name": "chương", "header_value": "5", "remainder": "Có indent"}, id="chuong_leading_whitespace"),
    pytest.param("Chương 1.", {"header_name": "chương", "header_value": "1", "remainder": ""}, id="chuong_1_dot_only"),
    pytest.param("Chương 2:", {"header_name": "chương", "header_value": "2", "remainder": ""}, id="chuong_2_colon_only"),
    pytest.param("Chương IX", {"header_name": "chương", "header_value": "IX", "remainder": ""}, id="chuong_IX_no_delimiter"),
    pytest.param("Chương 6- Phạm vi", {"header_name": "chương", "header_value": "6", "remainder": "Phạm vi"}, id="chuong_6_dash"),
    pytest.param("Chương XI. Mười một", {"header_name": "chương", "header_value": "XI", "remainder": "Mười một"}, id="chuong_roman_XI"),
    # False positive
    pytest.param("CHƯƠNG I CỦA ĐIỀU LỆ NHỮNG NGƯỜI", None, id="chuong_legacy_inline_title"),
    pytest.param("Chương 1 của phần trước", None, id="chuong_reference_cua"),
    pytest.param("Chương 2 quy định chi tiết", None, id="chuong_reference_quy_dinh"),
    pytest.param("Căn cứ Chương 1 Luật này", None, id="chuong_mid_sentence"),
    pytest.param("Chương", None, id="chuong_missing_index"),
    pytest.param("Chương abc không hợp lệ", None, id="chuong_invalid_index"),
    pytest.param("Không phải chương", None, id="chuong_plain_text"),
]

PHAN_CASES = [
    # Happy path — "Phần thứ" + một từ
    pytest.param("Phần thứ nhất. Quy định chung", {"header_name": "phần", "header_value": "nhất", "remainder": "Quy định chung"}, id="phan_thu_nhat_dot"),
    pytest.param("Phần thứ hai: Nghĩa vụ", {"header_name": "phần", "header_value": "hai", "remainder": "Nghĩa vụ"}, id="phan_thu_hai_colon"),
    pytest.param("Phần thứ ba. Chữ số", {"header_name": "phần", "header_value": "ba", "remainder": "Chữ số"}, id="phan_thu_ba"),
    pytest.param("Phần thứ I. La Mã", {"header_name": "phần", "header_value": "I", "remainder": "La Mã"}, id="phan_thu_roman_I"),
    pytest.param("Phần thứ 1. Số Arabic", {"header_name": "phần", "header_value": "1", "remainder": "Số Arabic"}, id="phan_thu_1_dot"),
    pytest.param("PHẦN THỨ TƯ. Viết hoa", {"header_name": "phần", "header_value": "TƯ", "remainder": "Viết hoa"}, id="phan_uppercase_keyword"),
    pytest.param("phần thứ năm. Chữ thường", {"header_name": "phần", "header_value": "năm", "remainder": "Chữ thường"}, id="phan_lowercase_keyword"),
    pytest.param("  Phần thứ sáu. Có indent", {"header_name": "phần", "header_value": "sáu", "remainder": "Có indent"}, id="phan_leading_whitespace"),
    pytest.param("Phần thứ bảy.", {"header_name": "phần", "header_value": "bảy", "remainder": ""}, id="phan_thu_bay_dot_only"),
    pytest.param("Phần thứ tám:", {"header_name": "phần", "header_value": "tám", "remainder": ""}, id="phan_thu_tam_colon_only"),
    pytest.param("Phần thứ chín", {"header_name": "phần", "header_value": "chín", "remainder": ""}, id="phan_thu_chin_no_delimiter"),
    pytest.param("Phần thứ mười- Phạm vi", {"header_name": "phần", "header_value": "mười", "remainder": "Phạm vi"}, id="phan_thu_muoi_dash"),
    pytest.param("Phần thứ mười một: Nội dung phần mười một", {"header_name": "phần", "header_value": "mười một", "remainder": "Nội dung phần mười một"}, id="phan_thu_muoi_mot_colon"),  # This case
    pytest.param("Phần thứ 2đ. Hậu tố đ", {"header_name": "phần", "header_value": "2đ", "remainder": "Hậu tố đ"}, id="phan_vn_suffix_in_word"),
    # False positive
    pytest.param("Phần 1. Không có chữ thứ", None, id="phan_missing_thu"),
    pytest.param("Phần thứ", None, id="phan_missing_label_word"),
    pytest.param("Phần thứ nhất của luật này", None, id="phan_reference_cua"),
    pytest.param("Phần thứ nhất quy định thêm", None, id="phan_reference_quy_dinh"),
    pytest.param("Theo phần thứ nhất", None, id="phan_mid_sentence"),
    pytest.param("Phần đầu văn bản", None, id="phan_prose_phan_dau"),
    pytest.param("Phần thứ nhất hai từ không gộp", None, id="phan_only_one_word_captured"),
    pytest.param("Không phải phần", None, id="phan_plain_text"),
]

@pytest.mark.parametrize("line,expected", DIEM_CASES)
def test_diem_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]

@pytest.mark.parametrize("line,expected", KHOAN_CASES)
def test_khoan_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]
        
@pytest.mark.parametrize("line,expected", DIEU_CASES)
def test_dieu_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]

@pytest.mark.parametrize("line,expected", TIEU_MUC_CASES)
def test_tieu_muc_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]

@pytest.mark.parametrize("line,expected", MUC_CASES)
def test_muc_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]

@pytest.mark.parametrize("line,expected", CHUONG_CASES)
def test_chuong_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]

@pytest.mark.parametrize("line,expected", PHAN_CASES)
def test_phan_header_pattern(line, expected):
    processor = VietnameseDocumentProcessor()
    header_name, header_value, remainder = processor._classify_level(line)
    if expected is None:
        assert header_name is None
    else:
        assert header_name == expected["header_name"]
        assert header_value == expected["header_value"]
        assert remainder == expected["remainder"]