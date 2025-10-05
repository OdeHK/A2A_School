from typing import TypedDict, Dict, Any, List, Optional
from services.rag.rag_service import RagService
from pydantic import Field
from services.models import PlanTaskOutput, PlanTaskOutputList, QuizQuestion, QuizQuestionOutput
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
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
    detail_table_of_contents: str
    user_request: str
    section_tasks: PlanTaskOutputList  # Tasks cho map-reduce
    generated_questions: QuizQuestionOutput  # Kết quả từ từng Generate node
    final_questions: List[Dict[str, Any]]  # Kết quả cuối cùng


class QuizGenerationService:
    """Main service điều phối việc sinh Quiz sử dụng LangGraph"""

    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        self.llm_service = rag_service.llm_service
        self.vector_service = rag_service.vector_service
        self.workflow = self._create_workflow()

    def generate_quiz_set(self, 
                        document_id: str,
                        user_request: str,
                        toc_data: str) -> str:
        """
        Main entry point để sinh bộ đề MCQ
        Args:
            document_id: ID của tài liệu tham khảo
            user_request: Yêu cầu của giáo viên
            toc_data: Dữ liệu mục lục chi tiết của tài liệu
        Returns:
            final_questions: Các câu hỏi được sinh ra ở dạng chuỗi
        """
        logger.info("========================================")
        logger.info("QUIZ GENERATION WORKFLOW START")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"User Request: {user_request}")
        logger.info("========================================")
        
        initial_state: QuizGenerationState = {
            "document_id": document_id,
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
            
            llm = self.rag_service.llm_service.llm
            
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
                    "toc": state["detail_table_of_contents"],
                    "request": state["user_request"],
                    "format_instructions": parser.get_format_instructions()
                })
                logger.info(f"Đã tạo được {len(section_tasks.tasks)} section tasks")
                
                # Generate log for created plan
                for i, task in enumerate(section_tasks.tasks):
                    logger.info(f"Task {i+1}: {task.section_title} - {task.number_of_questions} câu hỏi")
                
                if not section_tasks.tasks:
                    logger.warning("Không có section tasks nào được tạo, sử dụng fallback")
                    # Fallback với dummy data
                    section_tasks = PlanTaskOutputList(tasks=[
                        PlanTaskOutput(
                            section_id="section_1",
                            section_title="General Topics",
                            number_of_questions=3,
                            query_string="general concepts overview",
                            question_requirements="Focus on key concepts"
                        )
                    ])
                    
            except Exception as e:
                logger.error(f"Error in plan_node: {e}")
                section_tasks = PlanTaskOutputList(tasks=[
                    PlanTaskOutput(
                        section_id="section_1",
                        section_title="General Topics",
                        number_of_questions=3,
                        query_string="general concepts overview",
                        question_requirements="Focus on key concept"
                    )
                ])
            
            logger.info("=== PLAN NODE END ===")
            return {**state, "section_tasks": section_tasks}

        def map_generate_with_service(state: QuizGenerationState) -> Dict[str, Any]:
            """Map generate with access to rag_service"""
            logger.info("=== MAP GENERATE NODE START ===")
            section_tasks = state.get("section_tasks", PlanTaskOutputList(tasks=[]))
            logger.info(f"Số lượng sections cần xử lý: {len(section_tasks.tasks)}")
            generated_questions = QuizQuestionOutput(questions=[])
            
            for i, task in enumerate(section_tasks.tasks):
                logger.info(f"Đang xử lý section {i+1}/{len(section_tasks.tasks)}: {task.section_title}")
                logger.info(f"Section task details: {task}")

                try:
                    # Check if vectorstore is available
                    if (self.rag_service.vector_service.vectorstore is None):
                        logger.warning("Vectorstore not initialized, using dummy questions")
                        dummy_question = QuizQuestion(
                            type="multiple_choice",
                            title="Câu hỏi mẫu do chưa có dữ liệu",
                            options=["Tùy chọn A", "Tùy chọn B", "Tùy chọn C", "Tùy chọn D"],
                            answer="Tùy chọn A",
                            answer_explanation="Đây là câu hỏi mẫu do vectorstore chưa được khởi tạo"
                        )
                        generated_questions.questions.append(dummy_question)

                    else:
                        logger.info(f"Retrieving documents for query: {task.query_string}")
                        # TODO: có thể refract code vecto_service.py để thống nhất vector_service
                        retriever = self.rag_service.vector_service.vectorstore.as_retriever(
                            search_kwargs={"k": 5}
                        )
                        
                        relevant_docs = retriever.invoke(task.query_string)
                        logger.info(f"Retrieved {len(relevant_docs)} documents")
                        logger.info(f"Retrieved documents content: {[doc.page_content for doc in relevant_docs]}")

                        # Create Pydantic parser for quiz questions
                        quiz_parser = PydanticOutputParser(pydantic_object=QuizQuestionOutput)
                        
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
                        
                        logger.info(f"Generating {task.number_of_questions} questions using LLM")
                        prompt_input = {
                            "context": format_docs(relevant_docs),
                            "num_questions": task.number_of_questions,
                            "requirements": task.question_requirements,
                            "section_title": task.section_title,
                            "section_context": task.query_string,
                            "format_instructions": quiz_parser.get_format_instructions()
                        }
                        prompt_result = quiz_generation_prompt.invoke(prompt_input)
                        llm_result = self.rag_service.llm_service.llm.invoke(prompt_result)
                        logger.info(f"LLM raw output before parsing: {llm_result.__repr__()}")  # In ra dữ liệu thô

                        quiz_result = quiz_parser.invoke(llm_result)
                        logger.info(f"LLM raw output: {quiz_result}")

                        generated_questions.questions.extend(quiz_result.questions)
                        logger.info(f"Đã generate thành công {len(quiz_result.questions)} câu hỏi cho section '{task.section_title}'")
                        
                except Exception as e:
                    logger.error(f"Error generating questions for {task.section_title}: {e}")
                    # Create fallback question
                    fallback_question = QuizQuestion(
                        type="multiple_choice",
                        title=f"Câu hỏi về {task.section_title}",
                        options=["Lỗi khi tạo câu hỏi", "Vui lòng thử lại", "Không có dữ liệu", "Lỗi hệ thống"],
                        answer="Vui lòng thử lại",
                        answer_explanation=f"Có lỗi xảy ra khi tạo câu hỏi cho phần {task.section_title}"
                    )
                    generated_questions.questions.append(fallback_question)
                
            logger.info(f"MAP GENERATE hoàn thành: {len(generated_questions.questions)} câu hỏi")
            logger.info("=== MAP GENERATE NODE END ===")
            return {**state, "generated_questions": generated_questions}
        
        def aggregate_node(state: QuizGenerationState) -> Dict[str, Any]:
            """
            Node Aggregate: Tổng hợp kết quả từ tất cả Generate nodes
            """
            logger.info("=== AGGREGATE NODE START ===")
            generated_questions = state.get("generated_questions", QuizQuestionOutput(questions=[]))
            logger.info(f"Số lượng questions đã generate: {len(generated_questions.questions)}")
            
            # Convert to human-readable list
            final_questions = QuizGenerationService._convert_quiz_question_output_to_list(questions=generated_questions)
            
            # Write to file for record-keeping
            QuizGenerationService._write_questions_to_file(questions=generated_questions)
            logger.info("Written generated questions to file")

            
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
    def _write_questions_to_file(questions: QuizQuestionOutput):
        """Write generated questions to a JSON file for record-keeping"""
        # TODO: Đây cách tiếp cận tạm thời, cần cải thiện sau
        file_path = "session_data\\temp\\quiz_data.json" 

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(questions.model_dump(), f, ensure_ascii=False, indent=4)
        logger.info(f"Generated questions written to {file_path}")
