"""
MCP Server Configuration for Multi-Agent RAG System
Cấu hình MCP server cho hệ thống RAG đa agent
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

# MCP Server setup for connecting with agent cards
class MCPServerConfig:
    """Cấu hình MCP server cho các agents"""
    
    def __init__(self):
        self.agents = {
            "student_support": {
                "name": "Student Support Agent",
                "description": "Hỗ trợ học sinh với câu hỏi học thuật và lịch học",
                "capabilities": [
                    "answer_questions",
                    "recommend_materials", 
                    "set_reminders",
                    "study_planning"
                ],
                "endpoints": {
                    "ask_question": "/student/ask",
                    "get_recommendations": "/student/recommend",
                    "set_reminder": "/student/reminder",
                    "get_schedule": "/student/schedule"
                }
            },
            "teacher_support": {
                "name": "Teacher Support Agent",
                "description": "Hỗ trợ giáo viên với phương pháp dạy và quản lý lớp",
                "capabilities": [
                    "suggest_grouping",
                    "teaching_methods",
                    "teaching_materials",
                    "schedule_management"
                ],
                "endpoints": {
                    "group_students": "/teacher/group",
                    "suggest_method": "/teacher/method",
                    "get_materials": "/teacher/materials",
                    "manage_schedule": "/teacher/schedule"
                }
            },
            "data_analysis": {
                "name": "Data Analysis Agent", 
                "description": "Phân tích dữ liệu điểm số và xu hướng học tập",
                "capabilities": [
                    "analyze_performance",
                    "aggregate_scores",
                    "identify_at_risk",
                    "predict_trends"
                ],
                "endpoints": {
                    "analyze_class": "/analysis/class",
                    "aggregate_data": "/analysis/aggregate",
                    "find_at_risk": "/analysis/at-risk",
                    "predict_trends": "/analysis/trends"
                }
            }
        }
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Lấy cấu hình của một agent"""
        return self.agents.get(agent_name, {})
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Lấy cấu hình tất cả agents"""
        return self.agents
    
    def generate_mcp_manifest(self) -> Dict[str, Any]:
        """Tạo manifest file cho MCP"""
        manifest = {
            "schema_version": "1.0",
            "name": "School Management RAG System",
            "description": "Multi-agent RAG system for school management with 3 specialized agents",
            "version": "1.0.0",
            "agents": []
        }
        
        for agent_id, config in self.agents.items():
            agent_manifest = {
                "id": agent_id,
                "name": config["name"],
                "description": config["description"],
                "capabilities": config["capabilities"],
                "endpoints": config["endpoints"],
                "input_schema": self._get_input_schema(agent_id),
                "output_schema": self._get_output_schema(agent_id)
            }
            manifest["agents"].append(agent_manifest)
        
        return manifest
    
    def _get_input_schema(self, agent_id: str) -> Dict[str, Any]:
        """Định nghĩa schema đầu vào cho agent"""
        
        schemas = {
            "student_support": {
                "ask_question": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Câu hỏi của học sinh"},
                        "subject": {"type": "string", "description": "Môn học (tùy chọn)"},
                        "student_id": {"type": "string", "description": "ID học sinh"}
                    },
                    "required": ["question", "student_id"]
                },
                "get_recommendations": {
                    "type": "object", 
                    "properties": {
                        "subject": {"type": "string", "description": "Môn học"},
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                        "student_id": {"type": "string", "description": "ID học sinh"}
                    },
                    "required": ["subject", "student_id"]
                },
                "set_reminder": {
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "type": {"type": "string", "enum": ["class", "exam", "assignment"]},
                        "subject": {"type": "string"},
                        "datetime": {"type": "string", "format": "datetime"},
                        "note": {"type": "string"}
                    },
                    "required": ["student_id", "type", "subject", "datetime"]
                }
            },
            
            "teacher_support": {
                "group_students": {
                    "type": "object",
                    "properties": {
                        "student_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "student_id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "average_score": {"type": "number"},
                                    "learning_style": {"type": "string"}
                                }
                            }
                        },
                        "criteria": {"type": "string", "enum": ["academic_level", "learning_style"]}
                    },
                    "required": ["student_data"]
                },
                "suggest_method": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "class_level": {"type": "string"},
                        "topic": {"type": "string"}
                    },
                    "required": ["subject", "class_level", "topic"]
                }
            },
            
            "data_analysis": {
                "analyze_class": {
                    "type": "object",
                    "properties": {
                        "class_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "student_id": {"type": "string"},
                                    "subject": {"type": "string"},
                                    "score": {"type": "number"},
                                    "exam_date": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["class_data"]
                },
                "predict_trends": {
                    "type": "object",
                    "properties": {
                        "historical_data": {"type": "array"},
                        "prediction_period": {"type": "integer", "default": 6}
                    },
                    "required": ["historical_data"]
                }
            }
        }
        
        return schemas.get(agent_id, {})
    
    def _get_output_schema(self, agent_id: str) -> Dict[str, Any]:
        """Định nghĩa schema đầu ra cho agent"""
        
        base_response = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "message": {"type": "string"},
                "data": {"type": "object"},
                "timestamp": {"type": "string", "format": "datetime"}
            },
            "required": ["status", "message"]
        }
        
        return base_response

class MCPAgentHandler:
    """Handler để xử lý requests từ MCP clients"""
    
    def __init__(self, multi_agent_system):
        self.system = multi_agent_system
        self.config = MCPServerConfig()
    
    async def handle_student_request(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý request cho Student Support Agent"""
        try:
            agent = self.system.student_agent
            
            if action == "ask_question":
                response = agent.answer_academic_question(
                    data["question"], 
                    data.get("subject")
                )
                return self._format_response("success", "Câu trả lời", {"answer": response})
            
            elif action == "get_recommendations":
                response = agent.recommend_study_materials(
                    data["subject"],
                    data.get("difficulty", "medium")
                )
                return self._format_response("success", "Đề xuất tài liệu", response)
            
            elif action == "set_reminder":
                response = agent.set_study_reminder(
                    data["student_id"],
                    data["type"],
                    data["subject"],
                    data["datetime"],
                    data.get("note", "")
                )
                return self._format_response("success", "Đã thiết lập nhắc nhở", response)
            
            else:
                return self._format_response("error", f"Action không hỗ trợ: {action}")
        
        except Exception as e:
            return self._format_response("error", f"Lỗi xử lý request: {str(e)}")
    
    async def handle_teacher_request(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý request cho Teacher Support Agent"""
        try:
            agent = self.system.teacher_agent
            
            if action == "group_students":
                response = agent.suggest_student_grouping(
                    data["student_data"],
                    data.get("criteria", "academic_level")
                )
                return self._format_response("success", "Đề xuất chia nhóm", response)
            
            elif action == "suggest_method":
                response = agent.suggest_teaching_method(
                    data["subject"],
                    data["class_level"],
                    data["topic"]
                )
                return self._format_response("success", "Đề xuất phương pháp dạy", response)
            
            elif action == "get_materials":
                response = agent.suggest_teaching_materials(
                    data["subject"],
                    data["topic"],
                    data.get("material_type", "all")
                )
                return self._format_response("success", "Đề xuất tài liệu", response)
            
            else:
                return self._format_response("error", f"Action không hỗ trợ: {action}")
        
        except Exception as e:
            return self._format_response("error", f"Lỗi xử lý request: {str(e)}")
    
    async def handle_analysis_request(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý request cho Data Analysis Agent"""
        try:
            agent = self.system.data_agent
            
            if action == "analyze_class":
                response = agent.analyze_class_performance(data["class_data"])
                return self._format_response("success", "Phân tích lớp học", response)
            
            elif action == "aggregate_data":
                response = agent.aggregate_scores_by_criteria(
                    data["grade_data"],
                    data.get("group_by", "class")
                )
                return self._format_response("success", "Tổng hợp dữ liệu", response)
            
            elif action == "find_at_risk":
                response = agent.identify_students_need_support(
                    data["student_data"],
                    data.get("criteria")
                )
                return self._format_response("success", "Học sinh cần hỗ trợ", response)
            
            elif action == "predict_trends":
                response = agent.predict_learning_trends(
                    data["historical_data"],
                    data.get("prediction_period", 6)
                )
                return self._format_response("success", "Dự đoán xu hướng", response)
            
            else:
                return self._format_response("error", f"Action không hỗ trợ: {action}")
        
        except Exception as e:
            return self._format_response("error", f"Lỗi xử lý request: {str(e)}")
    
    def _format_response(self, status: str, message: str, data: Any = None) -> Dict[str, Any]:
        """Format chuẩn cho response"""
        response = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if data is not None:
            response["data"] = data
        
        return response

# Agent Card configurations
AGENT_CARDS = {
    "student_support_card": {
        "title": "🎓 Student Support Agent",
        "description": "Trợ lý học tập thông minh cho học sinh",
        "capabilities": [
            "💡 Trả lời câu hỏi học thuật (Toán, Lý, Hóa, Anh, Văn)",
            "📚 Đề xuất tài liệu học phù hợp theo trình độ",
            "⏰ Nhắc nhở lịch học và bài kiểm tra",
            "📝 Lập kế hoạch học tập cá nhân"
        ],
        "example_queries": [
            "Giải phương trình bậc hai x² + 5x + 6 = 0",
            "Định luật Newton thứ hai là gì?",
            "Đề xuất tài liệu ôn tập môn Toán nâng cao",
            "Nhắc tôi kiểm tra môn Lý vào thứ 6 tuần sau"
        ],
        "mcp_config": {
            "server_name": "student_support_server",
            "endpoints": ["ask", "recommend", "remind", "plan"]
        }
    },
    
    "teacher_support_card": {
        "title": "👨‍🏫 Teacher Support Agent", 
        "description": "Trợ lý sư phạm cho giáo viên",
        "capabilities": [
            "👥 Gợi ý chia nhóm học sinh theo trình độ/phong cách học",
            "📖 Đề xuất phương pháp giảng dạy hiệu quả",
            "📋 Cung cấp tài liệu giảng dạy và giáo án",
            "📅 Quản lý lịch dạy và nhắc nhở"
        ],
        "example_queries": [
            "Chia nhóm 30 học sinh theo trình độ học lực",
            "Phương pháp dạy hình học lớp 10 hiệu quả",
            "Tài liệu giảng dạy chủ đề ma trận",
            "Nhắc tôi chuẩn bị bài kiểm tra 15 phút"
        ],
        "mcp_config": {
            "server_name": "teacher_support_server", 
            "endpoints": ["group", "method", "materials", "schedule"]
        }
    },
    
    "data_analysis_card": {
        "title": "📊 Data Analysis Agent",
        "description": "Chuyên gia phân tích dữ liệu giáo dục",
        "capabilities": [
            "📈 Phân tích kết quả kiểm tra và học lực lớp",
            "📋 Tổng hợp điểm số theo lớp, môn, khối",
            "⚠️ Phát hiện học sinh cần hỗ trợ đặc biệt",
            "🔮 Dự đoán xu hướng học tập tương lai"
        ],
        "example_queries": [
            "Phân tích điểm kiểm tra toán của lớp 10A1",
            "Tổng hợp điểm trung bình theo khối lớp 10",
            "Học sinh nào cần hỗ trợ thêm?",
            "Xu hướng học lực lớp 10A2 trong 6 tháng tới"
        ],
        "mcp_config": {
            "server_name": "data_analysis_server",
            "endpoints": ["analyze", "aggregate", "at-risk", "trends"]
        }
    }
}

def generate_mcp_server_setup():
    """Tạo script setup MCP server"""
    
    setup_script = """
# MCP Server Setup Script
# Run this to setup MCP servers for the multi-agent RAG system

import asyncio
import json
from multi_agent_rag import MultiAgentRAGSystem
from mcp_config import MCPAgentHandler, AGENT_CARDS

async def setup_mcp_servers():
    \"\"\"Setup MCP servers for all agents\"\"\"
    
    # Initialize the multi-agent system
    # (You need to provide actual chunker, embedder, llm)
    system = MultiAgentRAGSystem(chunker, embedder, llm)
    handler = MCPAgentHandler(system)
    
    # Setup server configurations
    servers = {}
    
    for card_name, card_config in AGENT_CARDS.items():
        server_name = card_config["mcp_config"]["server_name"]
        servers[server_name] = {
            "handler": handler,
            "card_config": card_config,
            "status": "ready"
        }
    
    print("🚀 MCP Servers setup completed!")
    print(f"📊 Total servers: {len(servers)}")
    
    for server_name in servers:
        print(f"✅ {server_name} - Ready")
    
    return servers

# To run: python -c "from mcp_config import setup_mcp_servers; asyncio.run(setup_mcp_servers())"
"""
    
    return setup_script

def save_mcp_configuration():
    """Lưu cấu hình MCP ra file"""
    
    config = MCPServerConfig()
    manifest = config.generate_mcp_manifest()
    
    # Save manifest
    with open("mcp_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # Save agent cards
    with open("agent_cards.json", "w", encoding="utf-8") as f:
        json.dump(AGENT_CARDS, f, ensure_ascii=False, indent=2)
    
    # Save setup script
    with open("setup_mcp_servers.py", "w", encoding="utf-8") as f:
        f.write(generate_mcp_server_setup())
    
    print("✅ Đã lưu cấu hình MCP:")
    print("   - mcp_manifest.json")
    print("   - agent_cards.json") 
    print("   - setup_mcp_servers.py")

if __name__ == "__main__":
    save_mcp_configuration()
