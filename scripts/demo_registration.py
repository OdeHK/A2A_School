"""
Demo script - Test đầy đủ tính năng đăng ký
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.database_service import DatabaseService
from services.ui_integration_service import UIIntegrationService

def test_registration():
    """Test registration flow"""
    print("=" * 60)
    print("🧪 DEMO: TEST ĐĂNG KÝ USER")
    print("=" * 60)
    
    db_service = DatabaseService()
    ui_service = UIIntegrationService()
    
    # Test data
    test_users = [
        {
            "username": "teacher_demo",
            "password": "demo123456",
            "email": "teacher@demo.com",
            "full_name": "Giáo viên Demo"
        },
        {
            "username": "gv_khiem",
            "password": "khiem2024",
            "email": "khiem@school.edu.vn",
            "full_name": "Nguyễn Anh Khiêm"
        }
    ]
    
    for i, user_data in enumerate(test_users, 1):
        print(f"\n📝 Test case {i}: Đăng ký user '{user_data['username']}'")
        print("-" * 60)
        
        # Test via UIIntegrationService
        success, message = ui_service.register_user(
            username=user_data['username'],
            password=user_data['password'],
            email=user_data['email'],
            full_name=user_data['full_name']
        )
        
        if success:
            print(f"✅ {message}")
            print(f"   📧 Email: {user_data['email']}")
            print(f"   📛 Tên: {user_data['full_name']}")
            
            # Test authentication
            print(f"\n🔐 Kiểm tra đăng nhập...")
            is_auth = ui_service.authenticate_user(
                user_data['username'],
                user_data['password']
            )
            
            if is_auth:
                print(f"✅ Đăng nhập thành công với '{user_data['username']}'")
            else:
                print(f"❌ Không thể đăng nhập với '{user_data['username']}'")
        else:
            print(f"⚠️ {message}")
            if "đã tồn tại" in message:
                print(f"   ℹ️ User này đã được tạo trước đó")
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH DEMO")
    print("=" * 60)
    
    print("\n📊 Xem danh sách users trong database:")
    print("   python scripts/view_atlas_data.py")
    
    print("\n🚀 Đăng nhập vào ứng dụng:")
    print("   python ui/app.py")
    print("   → Chọn một trong các tài khoản đã tạo để đăng nhập")

if __name__ == "__main__":
    try:
        test_registration()
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
