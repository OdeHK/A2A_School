"""
Agent Service using LangGraph create_react_agent

This service provides a simpler approach using LangGraph's built-in 
create_react_agent with tools from existing services.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class AgentService:
    """
    Agent service using LangGraph's create_react_agent with tools.
    
    This provides a simpler, more efficient approach than custom tool management.
    """
    
    def __init__(
        self, 
        rag_service, 
        quiz_generation_service, 
        document_management_service,
        llm_service
    ):
        """
        Initialize the agent service with required services.
        
        Args:
            rag_service: RAG service for document queries
            quiz_generation_service: Service for quiz generation
            document_management_service: Service for document management
            llm_service: LLM service for the agent
        """
        self.rag_service = rag_service
        self.quiz_generation_service = quiz_generation_service
        self.document_management_service = document_management_service
        self.llm_service = llm_service
        
        # Create tools from services
        self.tools = self._create_tools()
        
        # Create the react agent
        self.agent = create_react_agent(
            model=llm_service.llm,
            tools=self.tools
        )
        
        logger.info(f"Agent service initialized with {len(self.tools)} tools")
    
    def _create_tools(self):
        """Create tools from the available services."""
        tools = []
        
        # RAG tool
        @tool
        def generate_rag_response(query: str) -> str:
            """
            Generate a response using RAG (Retrieval-Augmented Generation) based on uploaded documents.
            
            Args:
                query: The user's question or query to process using RAG
                
            Returns:
                The generated response from the RAG system
            """
            try:
                if not query or not query.strip():
                    return "Câu hỏi không hợp lệ. Vui lòng nhập lại."
                
                response = self.rag_service.generate_rag_response(query)
                return response
            except Exception as e:
                logger.error(f"Error in generate_rag_response tool: {str(e)}")
                return f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
        
        # Quiz generation tool
        @tool
        def generate_quiz_set(document_id: str, user_request: str) -> str:
            """
            Generate a quiz set based on uploaded documents and user requirements.
            
            Args:
                document_id: The document ID to generate quiz from
                user_request: User requirements and specifications for the quiz
                
            Returns:
                Information about the generated quiz set
            """
            try:
                if not document_id or not user_request:
                    return "Cần cung cấp document_id và yêu cầu người dùng."
                
                # Get table of contents
                toc_string = self.document_management_service.get_table_of_contents_as_string(document_id)
                
                if not toc_string:
                    return f"Không tìm thấy mục lục cho tài liệu: {document_id}"
                
                # Generate quiz
                result = self.quiz_generation_service.generate_quiz_set(
                    document_id=document_id,
                    user_request=user_request,
                    toc_data=toc_string
                )
                
                final_questions = result.get("final_questions", [])
                return f"✅ Đã tạo thành công {len(final_questions)} câu hỏi cho tài liệu {document_id}"
                
            except Exception as e:
                logger.error(f"Error in generate_quiz_set tool: {str(e)}")
                return f"Đã xảy ra lỗi khi tạo đề: {str(e)}"
        
        # Table of contents tool
        @tool
        def get_table_of_contents_as_string(document_id: str = "") -> str:
            """
            Get table of contents for a specific document formatted as string.
            If no document_id provided, list available documents.
            
            Args:
                document_id: The document ID to get table of contents for (optional)
                
            Returns:
                Table of contents formatted as string or list of documents
            """
            try:
                if not document_id:
                    # List available documents
                    documents = self.document_management_service.list_session_documents()
                    if not documents:
                        return "Chưa có tài liệu nào được tải lên."
                    
                    result = "📚 Danh sách tài liệu:\n"
                    for doc in documents:
                        result += f"- {doc.file_name} (ID: {doc.document_id})\n"
                    return result
                
                toc_string = self.document_management_service.get_table_of_contents_as_string(document_id)
                
                if toc_string:
                    return toc_string
                else:
                    return f"Không tìm thấy mục lục cho tài liệu: {document_id}"
                    
            except Exception as e:
                logger.error(f"Error in get_table_of_contents_as_string tool: {str(e)}")
                return f"Đã xảy ra lỗi: {str(e)}"
        
        tools.extend([generate_rag_response, generate_quiz_set, get_table_of_contents_as_string])
        return tools
    
    def handle_chat_query(self, query: str, chat_history: Optional[List] = None) -> Tuple[str, List]:
        """
        Handle chat query using the react agent.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            Tuple of (response, updated_chat_history)
        """
        try:
            if not query or not query.strip():
                return "Xin chào! Tôi có thể giúp gì cho bạn?", chat_history or []
            
            # Create message history for the agent
            messages = []
            
            # Add chat history if provided
            if chat_history:
                for user_msg, ai_msg in chat_history:
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=ai_msg))
            
            # Add current query
            messages.append(HumanMessage(content=query))
            
            # Invoke the agent
            result = self.agent.invoke({"messages": messages})
            
            logger.info(f"Result from agent: {result}")

            # Extract the response
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # Log intermediate messages if any
                for message in result["messages"]:
                    logger.info(f"Message: {message}")
            else:
                response = "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu."
            
            # Update chat history
            updated_history = chat_history or []
            updated_history.append((query, response))
            
            return response, updated_history
            
        except Exception as e:
            logger.error(f"Error in handle_chat_query: {str(e)}")
            error_response = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
            updated_history = chat_history or []
            updated_history.append((query, error_response))
            return error_response, updated_history
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "agent_initialized": self.agent is not None,
            "tools_count": len(self.tools),
            "available_tools": self.get_available_tools(),
            "rag_service_available": self.rag_service is not None,
            "quiz_service_available": self.quiz_generation_service is not None,
            "doc_service_available": self.document_management_service is not None
        }