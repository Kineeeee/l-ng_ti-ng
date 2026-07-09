import os
import sys
from dotenv import load_dotenv

# Thêm thư mục gốc vào path để có thể import các config nếu cần
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ LỖI: Không tìm thấy GEMINI_API_KEY trong file .env hoặc biến môi trường.")
        return False
    
    # In ra một phần API key để bảo mật nhưng vẫn kiểm tra được
    masked_key = f"{api_key[:8]}...{api_key[-8:]}" if len(api_key) > 16 else api_key
    print(f"🔑 Đang kiểm tra GEMINI_API_KEY: {masked_key}")
    
    # 1. Thử nghiệm với thư viện google-genai mới (dùng trong backend)
    print("\n--- PHẦN 1: Thử nghiệm bằng thư viện mới 'google-genai' ---")
    try:
        from google import genai
        from google.genai import types
        
        print("🔌 Khởi tạo google.genai Client...")
        client = genai.Client(api_key=api_key)
        
        # Thử nghiệm với các model khác nhau
        for model_name in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-3.5-flash']:
            print(f"🤖 Đang gửi request tới model '{model_name}'...")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Hãy phản hồi ngắn gọn: 'API Gemini hoạt động tốt!'",
                )
                print(f"✅ Kết nối thành công với model {model_name}!")
                print(f"💬 Phản hồi: {response.text.strip()}")
                print("🎉 Thư viện 'google-genai' hoạt động tốt.")
            except Exception as e:
                print(f"⚠️ Model '{model_name}' lỗi: {e}")
                
    except ImportError:
        print("❌ LỖI: Không thể import 'google-genai'. Thư viện này chưa được cài đặt trong môi trường này.")
        print("💡 Hãy chạy lệnh: pip install google-genai")
    except Exception as e:
        print(f"❌ LỖI tổng quát khi dùng google-genai: {e}")

    # 2. Thử nghiệm với thư viện google-generativeai cũ (nếu có trong requirements.txt)
    print("\n--- PHẦN 2: Thử nghiệm bằng thư viện cũ 'google-generativeai' ---")
    try:
        import google.generativeai as old_genai
        print("🔌 Khởi tạo google.generativeai...")
        old_genai.configure(api_key=api_key)
        
        for model_name in ['gemini-1.5-flash', 'gemini-pro']:
            print(f"🤖 Đang gửi request tới model '{model_name}'...")
            try:
                model = old_genai.GenerativeModel(model_name)
                response = model.generate_content("Hãy phản hồi ngắn gọn: 'API Gemini (cũ) hoạt động tốt!'")
                print(f"✅ Kết nối thành công với model {model_name}!")
                print(f"💬 Phản hồi: {response.text.strip()}")
                print("🎉 Thư viện 'google-generativeai' hoạt động tốt.")
                break
            except Exception as e:
                print(f"⚠️ Model '{model_name}' lỗi: {e}")
                
    except ImportError:
        print("❌ LỖI: Không thể import 'google-generativeai'.")
    except Exception as e:
        print(f"❌ LỖI tổng quát khi dùng google-generativeai: {e}")

if __name__ == "__main__":
    test_gemini()
