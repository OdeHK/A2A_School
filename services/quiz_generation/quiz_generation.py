from typing import TypedDict, Dict, Any, List, Optional, Tuple
from pydantic import Field
from services.rag.rag_service import RagService
from services.database_service import DatabaseService
from services.models import PlanTaskOutputList, QuizQuestion, QuizQuestionOutput, TableOfContentsSection
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
import json
import logging
import regex as re

logger = logging.getLogger(__name__)

def format_docs(documents) -> str:
    """Format retrieved documents cho LLM context"""
    if not documents:
        return ""
    return "\n\n".join([doc.page_content for doc in documents])

class CustomPydanticOutputParser(PydanticOutputParser):
    def parse(self, text: str) -> Any:
        # Check whether the text is wrapped in triple backticks
        if not (text.startswith("```") and text.endswith("```")):
            # If not, wrap it in triple backticks
            text = f"```json\n{text}\n```"

        # Add double backslashes in latex expressions
        text = re.sub(
            r'\$(.+?)\$',
            lambda m: "$" + m.group(1).replace("\\", "\\\\") + "$",
            text
        )
        return super().parse(text)

# ====== Graph State =========
class QuizGenerationState(TypedDict):
    document_id: str
    username: str  
    detail_table_of_contents: List[TableOfContentsSection]
    user_request: str
    section_tasks: PlanTaskOutputList  # Tasks cho map-reduce
    generated_questions: QuizQuestionOutput  # Kết quả từ từng Generate node
    final_questions: List[Dict[str, Any]]  # Kết quả cuối cùng


class QuizGenerationService:
    """Main service điều phối việc sinh Quiz sử dụng LangGraph"""

    def __init__(self, rag_service: RagService, database_service: DatabaseService):
        self.rag_service = rag_service
        self.llm_service = rag_service.llm_service
        self.vector_service = rag_service.vector_service
        self.database_service = database_service
        self.workflow = self._create_workflow()

    def generate_quiz_set(self, 
                        document_id: str,
                        username: str,
                        user_request: str,
                        toc_data: List[TableOfContentsSection]) -> str:
        """
        Main entry point để sinh bộ đề MCQ
        Args:
            document_id: ID của tài liệu tham khảo
            username: Username for metadata filtering
            user_request: Yêu cầu của giáo viên
            toc_data: Dữ liệu mục lục chi tiết của tài liệu dạng List[TableOfContentsSection]
        Returns:
            final_questions: Các câu hỏi được sinh ra ở dạng chuỗi
        """
        logger.info("========================================")
        logger.info("QUIZ GENERATION WORKFLOW START")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Username: {username}")
        logger.info(f"User Request: {user_request}")
        logger.info("========================================")
        
        initial_state: QuizGenerationState = {
            "document_id": document_id,
            "username": username,
            "detail_table_of_contents": toc_data,
            "user_request": user_request,
            "section_tasks": PlanTaskOutputList(tasks=[]),
            "generated_questions": QuizQuestionOutput(questions=[]),
            "final_questions": []
        }
        
        result = self.workflow.invoke(initial_state)
        
        logger.info("========================================")
        logger.info("QUIZ GENERATION WORKFLOW COMPLETE")
        logger.info(f"Total final questions: {len(result.get('generated_questions', QuizQuestionOutput(questions=[])).questions)}")
        logger.info("========================================")
        
        return result.get("final_questions", [])

    def _create_workflow(self):
        """Create LangGraph workflow with properly configured nodes"""

        def plan_node_with_service(state: QuizGenerationState) -> Dict[str, Any]:
            """Plan node with access to rag_service"""
            logger.info("=== PLAN NODE START ===")
            logger.info(f"Document ID: {state['document_id']}")
            logger.info(f"User request: {state['user_request']}")
            
            toc_data = state["detail_table_of_contents"]
            llm = self.rag_service.llm_service.llm
            
            # Convert ToC sections to string for LLM processing
            toc_string = QuizGenerationService._convert_toc_to_string(toc_data)
            logger.info(f"Converted ToC to string format with {len(toc_data)} top-level sections")

            # Create Pydantic parser for structured output
            parser = PydanticOutputParser(pydantic_object=PlanTaskOutputList)

            plan_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "Reasoning: Medium"
                 "Your task is to design a question distribution plan for an exam set. "
                 "You are not generating the actual questions, only planning how the knowledge should be allocated across the test."
                ),
                ("human", 
                 "Input: You will be provided with:\n"
                 "The Table of Contents from the textbook or curriculum, organized hierarchically.\n"
                 "The teacher’s requirements regarding exam content, such as preferred question types, target audience, or emphasis on specific topics.\n"
                 "Output: Return a list of tasks in the form of a JSON object. Each task corresponds to a lowest-level section (leaf node) from the Table of Contents. "
                 "The total number of questions across all tasks must match the overall exam question count.\n"
                 "List at most 3 tasks. Select the 3 most important sections. Do not include any task with number_of_questions = 0."
                 "Each task must include the following fields:\n"
                 "section_id (string): A unique identifier for the section\n"
                 "section_title (string): The official title of the section as listed in the Table of Contents\n"
                 "number_of_questions (positive int): The number of questions allocated to this section. This should reflect the importance or emphasis based on teacher input and curriculum weight.\n"
                 "question_requirements (string): A brief description of the expected question format and audience. This is derived from the teacher’s instructions. Default (if unspecified): \"Multiple choice questions with 4 options, containing 1 correct answer, designed for university-level students.\"\n"
                 "query_string (string): Using the section title and its immediate parent section title from the Table of Contents, write one descriptive sentence that explains the context and focus of this section. The sentence should reflect the hierarchical structure of the curriculum and highlight key concepts or themes relevant to the section. The sentence must be written in the same language used in the Table of Contents\n\n"
                 "Format output instruction:\n {format_instructions}\n\n"
                 "# Teacher's requirements\n"
                 "{request}\n\n"
                 "# Table of content\n"
                 "{toc}"
                )
            ])
            try:
                # Create chain with parser
                chain = plan_prompt | llm | parser
                section_tasks = chain.invoke({
                    "toc": toc_string,
                    "request": state["user_request"],
                    "format_instructions": parser.get_format_instructions()
                })
                logger.info(f"Đã tạo được {len(section_tasks.tasks)} section tasks")
                
                # Generate log for created plan
                for i, task in enumerate(section_tasks.tasks):
                    logger.info(f"Task {i+1}: {task.section_title} - {task.number_of_questions} câu hỏi")
                
                if not section_tasks.tasks:
                    logger.warning("Không có section tasks nào được tạo, sử dụng fallback")
                    section_tasks = PlanTaskOutputList(tasks=[])
                    
            except Exception as e:
                logger.error(f"Error in plan_node: {e}")
                section_tasks = PlanTaskOutputList(tasks=[])

            logger.info("=== PLAN NODE END ===")
            return {**state, "section_tasks": section_tasks}

        def map_generate_with_service(state: QuizGenerationState) -> Dict[str, Any]:
            """Map generate with access to rag_service and metadata filtering"""
            logger.info("=== MAP GENERATE NODE START ===")
            section_tasks = state.get("section_tasks", PlanTaskOutputList(tasks=[]))
            toc_data = state.get("detail_table_of_contents")
            document_id = state.get("document_id")
            username = state.get("username")
            
            logger.info(f"Số lượng sections cần xử lý: {len(section_tasks.tasks)}")
            logger.info(f"Document ID: {document_id}, Username: {username}")
            
            generated_questions = QuizQuestionOutput(questions=[])
            
            # Check if vectorstore is available
            if self.rag_service.vector_service.vectorstore is None:
                logger.warning("Vectorstore not initialized, cannot generate questions")
                logger.info("=== MAP GENERATE NODE END ===")
                return {**state, "generated_questions": generated_questions}
            
            # Create Pydantic parser for quiz questions 
            quiz_parser = PydanticOutputParser(pydantic_object=QuizQuestionOutput)
            
            # Create prompt template
            quiz_generation_prompt = ChatPromptTemplate.from_messages([
                ("system", "Reasoning: Low. Act as a teacher responsible for assessing students' understanding. Your task is to generate exam questions based on the user's intent and the provided textbook content."),
                ("human", "Instructions\n"
                          "You are required to generate a quiz set with {num_questions} questions for the section titled '{section_title}' from the textbook. The SECTION_CONTEXT provides background information to help you understand the role and scope of this section within the overall curriculum.\n"
                          "Relevant content for this section is provided in the RETRIEVED_CONTEXT.\n\n"
                          "Stick strictly to the RETRIEVED_CONTEXT. Do not introduce any new information or assumptions beyond what is provided.\n\n"
                          "Question Requirements:\n"
                          "{requirements}\n\n"
                          "Response Formats:\n {format_instructions}\n"
                          "Math formatting: For inline mathematical expressions, enclose them in single dollar signs: $...$. For block equations, enclose them in double dollar signs: $$...$$\n"
                          "Your response must be written in Vietnamese\n"
                          "SECTION_CONTEXT:\n"
                          "{section_context}\n"
                          "RETRIEVED_CONTEXT:\n"
                          "{context}\n\n")
            ])
            
            # Step 1: Collect all prompt inputs for batch processing
            batch_prompt_inputs = []
            task_info_list = []  # Keep track of task information for logging
            
            for i, task in enumerate(section_tasks.tasks):
                logger.info(f"Preparing batch input {i+1}/{len(section_tasks.tasks)}: {task.section_title}")
                logger.info(f"Section task details: {task}")
                
                try:
                    # Get page range for the section to filter retrieved documents
                    start_page, end_page = self._get_page_range_from_section(toc_data, task.section_title)
                    
                    logger.info(f"Retrieving documents for query: {task.query_string}")
                    # Prepare metadata filter for document-specific and user-specific queries
                    metadata_filter = {
                        "$and": [
                            {"document_id": document_id},
                            {"username": username},
                            {"page_number": {"$gte": start_page} if start_page is not None else {}},
                            {"page_number": {"$lte": end_page} if end_page is not None else {}}
                        ]
                    }
                    logger.info(f"Applying metadata filter: {metadata_filter}")
                    
                    # Use retrieve_documents with metadata filtering
                    relevant_docs = self.rag_service.retrieve_documents(
                        query=task.query_string,
                        top_k=end_page - start_page + 1 if start_page is not None and end_page is not None else 5,
                        filter=metadata_filter
                    )
                    
                    logger.info(f"Retrieved {len(relevant_docs)} documents")
                    logger.debug(f"Retrieved documents content: {[doc.page_content for doc in relevant_docs]}")
                    
                    # Prepare prompt input for this task
                    prompt_input = {
                        "context": format_docs(relevant_docs),
                        "num_questions": task.number_of_questions,
                        "requirements": task.question_requirements,
                        "section_title": task.section_title,
                        "section_context": task.query_string,
                        "format_instructions": quiz_parser.get_format_instructions()
                    }
                    
                    batch_prompt_inputs.append(prompt_input)
                    task_info_list.append(task)
                    
                except Exception as e:
                    logger.error(f"Error preparing batch input for {task.section_title}: {e}")
            
            # Step 2: Batch invoke LLM if we have valid inputs
            if batch_prompt_inputs:
                try:
                    logger.info(f"Batch invoking LLM with {len(batch_prompt_inputs)} prompts")
                    
                    # Create chain with parser
                    chain = quiz_generation_prompt | self.rag_service.llm_service.llm | quiz_parser
                    
                    # Batch invoke
                    batch_results = chain.batch(batch_prompt_inputs)
                    
                    logger.info(f"Batch invoke completed, processing {len(batch_results)} results")
                    
                    # Step 3: Process batch results
                    for i, (quiz_result, task) in enumerate(zip(batch_results, task_info_list)):
                        logger.info(f"Processing result {i+1}/{len(batch_results)} for section: {task.section_title}")
                        logger.info(f"LLM output: {quiz_result}")
                        
                        generated_questions.questions.extend(quiz_result.questions)
                        logger.info(f"Đã generate thành công {len(quiz_result.questions)} câu hỏi cho section '{task.section_title}'")
                        
                except Exception as e:
                    logger.error(f"Error during batch LLM invocation: {e}")
            else:
                logger.warning("No valid batch inputs prepared, skipping LLM invocation")
            
            logger.info(f"MAP GENERATE hoàn thành: {len(generated_questions.questions)} câu hỏi")
            logger.info("=== MAP GENERATE NODE END ===")
            return {**state, "generated_questions": generated_questions}
        
        def aggregate_node(state: QuizGenerationState) -> Dict[str, Any]:
            """
            Node Aggregate: Tổng hợp kết quả từ tất cả Generate nodes
            """
            logger.info("=== AGGREGATE NODE START ===")
            username = state.get("username")
            generated_questions = state.get("generated_questions", QuizQuestionOutput(questions=[]))
            logger.info(f"Số lượng questions đã generate: {len(generated_questions.questions)}")
            
            if generated_questions.questions:
                # Convert to human-readable list
                final_questions = QuizGenerationService._convert_quiz_question_output_to_list(questions=generated_questions)
            
                # Write to file for record-keeping
                self._write_questions_to_database(username=username, questions=generated_questions)
            else:
                final_questions = "Hiện tại không có câu hỏi nào được tạo ra. Bạn có thể thử lại hoặc điều chỉnh yêu cầu"
            
            logger.info("=== AGGREGATE NODE END ===")
            return {
                **state,
                "final_questions": final_questions,
            }

        workflow = StateGraph(QuizGenerationState)
        
        # Add nodes với closures
        workflow.add_node("plan", plan_node_with_service)
        workflow.add_node("map_generate", map_generate_with_service)  
        workflow.add_node("aggregate", aggregate_node)
        
        # Define flow
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "map_generate")
        workflow.add_edge("map_generate", "aggregate")
        workflow.add_edge("aggregate", END)
        
        return workflow.compile()
    
    @staticmethod
    def _convert_quiz_question_output_to_list(questions: QuizQuestionOutput):
        """Convert QuizQuestionOutput to a human-readable string list
        Example output:
        1. What is the capital of France?
            A. Berlin
            B. Madrid
            C. Paris
            D. Rome
        
        """
        str_output = ""
        for idx, question in enumerate(questions.questions):
            str_output += f"{idx+1}. {question.title}"

            # Add options for multiple choice questions
            if question.type == "multiple_choice" and question.options:
                for opt_idx, option in enumerate(question.options):
                    str_output += f"\n   {chr(65 + opt_idx)}. {option}"
            
            # Add answer and explanation
            if question.type == "multiple_choice" and question.answer and question.answer_explanation:
                str_output += f"\n   -> {question.answer}: {question.answer_explanation}"
            str_output += "\n"

        return str_output
    
    @staticmethod
    def _convert_toc_to_string(toc_sections: List[TableOfContentsSection], indent_level: int = 0) -> str:
        """
        Convert Table of Contents sections to a formatted string representation.
        
        Args:
            toc_sections: List of TableOfContentsSection objects
            indent_level: Current indentation level for hierarchical display
            
        Returns:
            Formatted string representation of the table of contents
        """
        result = []
        indent = "  " * indent_level
        
        for section in toc_sections:
            # Format section with ID, title, and page number if available
            page_info = f" (Page {section.page_number})" if section.page_number else ""
            section_line = f"{indent}[{section.section_id}] {section.section_title}{page_info}"
            result.append(section_line)
            
            # Recursively process children
            if section.children:
                child_str = QuizGenerationService._convert_toc_to_string(
                    section.children, 
                    indent_level + 1
                )
                result.append(child_str)
        
        return "\n".join(result)

    def _write_questions_to_database(self, username: str, questions: QuizQuestionOutput):
        """Write generated questions to database"""
        self.database_service.save_quizset(username=username, quizset=questions)
    
    def _get_page_range_from_section(
        self, 
        toc_sections: List[TableOfContentsSection], 
        section_title: str
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Get page range for a given section by searching through the TOC structure.
        
        Args:
            toc_sections: List of TableOfContentsSection objects to search through
            section_title: Title of the section to find
            
        Returns:
            Tuple of (start_page, end_page). Returns (None, None) if section not found
            or if page information is not available.
        """
        def search_section(sections: List[TableOfContentsSection]) -> tuple[Optional[int], Optional[int]]:
            """Recursively search for section by title"""
            for section in sections:
                # Check if this is the target section
                if section.section_title == section_title:
                    page_number = section.page_number
                    end_page = section.end_page
                    logger.info(f"Section '{section_title}' found: start_page={page_number}, end_page={end_page}")
                    return (page_number, end_page)
      
                
                # Search in children recursively
                if section.children:
                    result = search_section(section.children)
                    if result != (None, None):
                        return result
            
            return (None, None)
        
        # Perform the search
        logger.info(f"Searching for section '{section_title}' in TOC")
        start_page, end_page = search_section(toc_sections)
        
        return (start_page, end_page)