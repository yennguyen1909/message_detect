import google.generativeai as genai
import json
import pandas as pd

GOOGLE_API_KEY = "AIzaSyBN_b1WG2OZMHV4qin76DLo0JLAW2D91LE"
genai.configure(api_key=GOOGLE_API_KEY)

def call_gemini_for_product_detection(prompt, input_path):
    """
    Gọi API của Gemini để trích xuất thông tin sản phẩm từ tin nhắn.

    Args:
        prompt (str): Prompt để hướng dẫn Gemini.
        input_data (list): Danh sách các dictionary chứa 'shop_order' và 'clean_content'.

    Returns:
        list: Danh sách các JSON chứa thông tin sản phẩm đã được trích xuất.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    extracted_data_with_original_content = []

    for item in input_path:
        shop_order = item['shop_order']
        clean_content_original = item['clean_content']
        prompt_with_context = f"""{prompt}

Input:
shop_order: {shop_order}
clean_content: {clean_content_original}
"""
        try:
            response = model.generate_content(prompt_with_context)
            if response.text:
                try:
                    extracted_info_list = json.loads(response.text)
                    # Không cần thêm trường 'clean_content' thủ công nữa
                    extracted_data_with_original_content.extend(extracted_info_list)
                except json.JSONDecodeError:
                    print(f"Lỗi giải mã JSON cho shop_order: {shop_order}")
                    print(f"Nội dung phản hồi: {response.text}")
            else:
                print(f"Không có phản hồi từ Gemini cho shop_order: {shop_order}")
        except Exception as e:
            print(f"Lỗi khi gọi API Gemini cho shop_order {shop_order}: {e}")

    return extracted_data_with_original_content

def chunk_csv(file_path, chunk_size=2):
    """
    Đọc file CSV và chia thành các chunk có kích thước chunk_size.

    Args:
        file_path (str): Đường dẫn đến file CSV.
        chunk_size (int): Số dòng trong mỗi chunk.

    Yields:
        pandas.DataFrame: Một chunk dữ liệu từ file CSV.
    """
    reader = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in reader:
        yield chunk

if __name__ == "__main__":
    prompt = """
Role: Bạn là hệ thống chăm sóc khách hàng của Giao Hàng Tiết Kiệm (GHTK)
Goal:
Bạn đang hỗ trợ user về xử lý yêu cầu trích xuất thông tin về sản phẩm (3 levels) và thuộc tính sản phẩm. Đối với các yêu cầu có cùng một loại sản phẩm nhưng khác nhau về thuộc tính và có số lượng tương ứng cho từng thuộc tính, hãy tách thành các đối tượng JSON riêng biệt.
Nhận input là lịch sử đoạn chat.
Nhiệm vụ của bạn là trích xuất các tham số (nếu có) của yêu cầu hiện tại, nếu tham số là thời gian thì chuẩn hoá giá trị thời gian ra thành định dạng<ctrl3348>-MM-DD HH:MM:SS bao gồm :
<param>
[
  {
    "message":{"description":"Tin nhắn gốc từ shop hoặc khách hàng, là nội dung trường content", "type":"string"}
  },
  {
    "products": {
      "description": "Chi tiết các sản phẩm khác nhau được yêu cầu, đặc biệt khi có sự khác biệt về thuộc tính và số lượng.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "product_name": {"type": "string"},
          "quantity": {"type": "integer"},
          "lv1": {"type": "array", "items": {"type": "string"}},
          "lv2": {"type": "array", "items": {"type": "string"}},
          "lv3": {"type": "array", "items": {"type": "string"}},
          "attribute": {
            "type": "object",
            "properties": {
              "attribute": {"type": "string"},
              "value": {"type": "string"}
            }
          },
          "attributes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "attribute": {"type": "string"},
                "value": {"type": "string"}
              }
            }
          }
        }
      }
    }
  }
]
</param>
Constraints:
Kết quả trả về ở dạng json list.
Chỉ trả về tham số, không giải thích gì thêm.
Chỉ trích xuất duy nhất 1 mã đơn của yêu cầu mà user đang quan tâm hiện tại.
Nếu có các sản phẩm giống nhau nhưng khác nhau về thuộc tính và có số lượng tương ứng cho từng nhóm thuộc tính, hãy tách thành các đối tượng JSON riêng biệt trong mảng products. Sử dụng trường attribute nếu chỉ có một thuộc tính, và attributes nếu có từ hai thuộc tính trở lên cho mỗi sản phẩm trong mảng products.
"""

    file_path = "D:/GHTK/smarttagger_shops/message_22198578_sample1000.csv" 
    chunk_size = 2
    all_extracted_data = []
    
    for i, chunk in enumerate(chunk_csv(file_path, chunk_size)):
        print(f"Processing chunk {i+1}:")
        input_chunk_data = chunk.to_dict('records')
        extracted_chunk_data = call_gemini_for_product_detection(prompt, input_chunk_data)
        all_extracted_data.extend(extracted_chunk_data)
        print(f"Extracted {len(extracted_chunk_data)} items from chunk {i+1}")
        print("-" * 20)

    print("Final extracted data:")
    print(json.dumps(all_extracted_data, indent=2, ensure_ascii=False))

    
