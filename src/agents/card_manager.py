# src/agents/card_manager.py
# Module quản lý việc tải, truy cập và thực thi các "Agent Card".

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Sử dụng relative import để giữ cấu trúc module
from ..core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

@dataclass
class AgentCard:
    """
    Một cấu trúc dữ liệu (dataclass) để biểu diễn một Agent Card.
    Nó chứa tất cả thông tin cần thiết về một "kỹ năng" của AI,
    từ metadata cho đến các mẫu prompt để thực thi.
    """
    id: str
    name: str
    description: str
    category: str
    prompts: Dict[str, str] = field(default_factory=dict)
    parameters: List[Dict[str, str]] = field(default_factory=list)
    
    # Kiểm tra sau khi khởi tạo để đảm bảo card có đủ thông tin cần thiết
    def __post_init__(self):
        if not self.id or not self.name or not self.prompts.get("user_template"):
            raise ValueError("Agent Card thiếu các trường bắt buộc: id, name, prompts.user_template")

class CardManager:
    """
    Quản lý vòng đời của các Agent Card. Chịu trách nhiệm tải các định nghĩa card
    từ file, cung cấp thông tin và điều phối việc thực thi chúng.
    """
    def __init__(self, cards_path: str, llm_provider: LLMProvider):
        """
        Khởi tạo CardManager.
        
        Args:
            cards_path (str): Đường dẫn đến thư mục chứa các file JSON định nghĩa card.
            llm_provider (LLMProvider): Instance của LLM provider để thực thi card.
        """
        # Áp dụng Dependency Injection
        self.cards_path = Path(cards_path)
        self.llm_provider = llm_provider
        
        # Dictionary để lưu các card đã được tải vào bộ nhớ
        self.cards: Dict[str, AgentCard] = {}
        
        # Tự động tải tất cả các card khi khởi tạo
        self._load_cards()
        logger.info(f"✅ CardManager đã được khởi tạo và tải {len(self.cards)} card.")

    def _load_cards(self):
        """
        Quét thư mục cards_path, đọc từng file .json, parse và tạo đối tượng AgentCard.
        Đây là cơ chế giúp hệ thống có khả năng mở rộng "plug-and-play".
        """
        if not self.cards_path.is_dir():
            logger.warning(f"Thư mục Agent Cards '{self.cards_path}' không tồn tại. Bỏ qua việc tải card.")
            return

        logger.info(f"Đang tải các Agent Card từ: {self.cards_path}")
        for card_file in self.cards_path.glob("*.json"):
            try:
                with open(card_file, 'r', encoding='utf-8') as f:
                    card_data = json.load(f)
                    # Sử dụng ** để giải nén dictionary thành các tham số cho constructor
                    card = AgentCard(**card_data)
                    self.cards[card.id] = card
                    logger.info(f"  -> Đã tải thành công card: '{card.name}' (ID: {card.id})")
            except json.JSONDecodeError:
                logger.error(f"Lỗi parse JSON trong file card: {card_file}")
            except (ValueError, TypeError) as e:
                logger.error(f"Dữ liệu trong file card '{card_file}' không hợp lệ: {e}")
            except Exception as e:
                logger.error(f"Lỗi không xác định khi tải file card '{card_file}': {e}")
    
    def get_card(self, card_id: str) -> Optional[AgentCard]:
        """Lấy một đối tượng AgentCard bằng ID của nó."""
        return self.cards.get(card_id)

    def get_all_cards_info(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách thông tin tóm tắt của tất cả các card,
        thường dùng để hiển thị trên giao diện người dùng (UI).
        """
        return [
            {
                "id": card.id,
                "name": card.name,
                "description": card.description,
                "category": card.category
            } 
            for card in self.cards.values()
        ]

    def execute_card(self, card_id: str, context: Dict[str, Any]) -> str:
        """
        Hàm thực thi chính.
        Nó lấy một card, xây dựng prompt cuối cùng và gọi LLM.
        """
        logger.info(f"Bắt đầu thực thi card '{card_id}'...")
        card = self.get_card(card_id)
        if not card:
            logger.error(f"Không tìm thấy card với ID: {card_id}")
            return f"Lỗi: Không tìm thấy Agent Card '{card_id}'."

        # Lấy các mẫu prompt từ định nghĩa của card
        system_prompt = card.prompts.get("system", "") # Prompt hệ thống (vai trò của AI)
        user_template = card.prompts.get("user_template", "") # Mẫu prompt của người dùng

        try:
            # Dùng .format(**context) để điền các giá trị từ context vào template
            # Ví dụ: nếu template là "Tóm tắt đoạn văn sau: {text}" và context là {"text": "abc"}
            # thì user_prompt sẽ là "Tóm tắt đoạn văn sau: abc"
            user_prompt = user_template.format(**context)
        except KeyError as e:
            logger.error(f"Lỗi khi format prompt cho card '{card_id}'. Context thiếu key: {e}")
            return f"Lỗi: Đầu vào cho card '{card.name}' không đầy đủ. Thiếu thông tin: {e}"

        # Gọi LLM Provider để nhận kết quả
        response = self.llm_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        logger.info(f"Thực thi card '{card_id}' thành công.")
        return response

# --- Hướng dẫn tạo một file Agent Card ---
#
# Để thêm một "kỹ năng" mới cho AI, bạn chỉ cần tạo một file .json trong thư mục
# được chỉ định trong config (ví dụ: data/agent_cards/) với cấu trúc sau:
#
# Ví dụ file: summarize_text.json
# {
#     "id": "summarize_document",
#     "name": "Tóm tắt Văn bản",
#     "description": "Tóm tắt một đoạn văn bản dài thành các ý chính ngắn gọn.",
#     "category": "Xử lý Văn bản",
#     "prompts": {
#         "system": "Bạn là một trợ lý AI chuyên về tóm tắt văn bản. Hãy tóm tắt một cách súc tích, chính xác và dễ hiểu.",
#         "user_template": "Dựa vào đoạn văn bản sau đây, hãy tạo ra một bản tóm tắt khoảng 3-5 câu:\n\n---VĂN BẢN---\n{document_text}\n\n---TÓM TẮT---"
#     },
#     "parameters": [
#         {
#             "name": "document_text",
#             "description": "Đoạn văn bản cần được tóm tắt."
#         }
#     ]
# }
#
# CardManager sẽ tự động nhận diện và tải file này khi khởi động.

