"""
Script để tạo user mới trong MongoDB
Usage: 
    python scripts/create_user.py
    python scripts/create_user.py --username teacher1 --password teacher123
    python scripts/create_user.py --username teacher1 --password teacher123 --email teacher1@school.edu.vn --full_name "Nguyễn Văn A"
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.database_service import DatabaseService
import argparse

def create_user(username: str, password: str, email: str = None, full_name: str = None):
    """Create a new user in MongoDB"""
    db_service = DatabaseService()
    
    try:
        print(f"🔄 Đang tạo user '{username}'...")
        
        # Create user
        success, message = db_service.create_user(
            username=username, 
            password=password,
            email=email,
            full_name=full_name
        )
        
        if success:
            print(f"✅ {message}")
            print(f"\n📊 Thông tin tài khoản:")
            print(f"   👤 Username: {username}")
            print(f"   🔒 Password: {password}")
            if email:
                print(f"   📧 Email: {email}")
            if full_name:
                print(f"   📛 Họ tên: {full_name}")
            
            # Verify authentication
            print(f"\n🔐 Đang kiểm tra đăng nhập...")
            is_auth = db_service.authenticate_user(username, password)
            if is_auth:
                print(f"✅ Xác thực thành công! Có thể đăng nhập với tài khoản này.")
            else:
                print(f"⚠️ Không thể xác thực tài khoản vừa tạo.")
            
            return True
        else:
            print(f"❌ {message}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Tạo tài khoản mới trong hệ thống',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tạo user cơ bản
  python scripts/create_user.py --username demo --password demo123
  
  # Tạo user với đầy đủ thông tin
  python scripts/create_user.py --username teacher1 --password teacher123 \\
      --email teacher1@school.edu.vn --full_name "Nguyễn Văn A"
        """
    )
    
    parser.add_argument('--username', type=str, default='demo', 
                       help='Tên đăng nhập (ít nhất 3 ký tự)')
    parser.add_argument('--password', type=str, default='demo123', 
                       help='Mật khẩu (ít nhất 6 ký tự)')
    parser.add_argument('--email', type=str, default=None,
                       help='Địa chỉ email (tùy chọn)')
    parser.add_argument('--full_name', type=str, default=None,
                       help='Họ và tên đầy đủ (tùy chọn)')
    
    args = parser.parse_args()
    
    # Validation
    if len(args.username) < 3:
        print("❌ Username phải có ít nhất 3 ký tự")
        sys.exit(1)
    
    if len(args.password) < 6:
        print("❌ Password phải có ít nhất 6 ký tự")
        sys.exit(1)
    
    # Create user
    success = create_user(
        username=args.username,
        password=args.password,
        email=args.email,
        full_name=args.full_name
    )
    
    if success:
        print("\n✅ Hoàn tất! Bạn có thể đăng nhập vào hệ thống với tài khoản này.")
        sys.exit(0)
    else:
        print("\n❌ Không thể tạo tài khoản.")
        sys.exit(1)
