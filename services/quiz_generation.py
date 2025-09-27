from typing import TypedDict, Dict, Any, List
from services.rag.rag_service import RagService
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain.output_parsers import PydanticOutputParser
import json
import logging

logger = logging.getLogger(__name__)

class QuizGenerationState(TypedDict):
    document_id: str
    detail_table_of_contents: str
    user_request: str
    section_tasks: List[Dict[str, Any]]  # Tasks cho map-reduce
    generated_questions: List[Dict[str, Any]]  # Kết quả từ từng Generate node
    final_questions: List[Dict[str, Any]]  # Kết quả cuối cùng


def format_docs(documents) -> str:
    """Format retrieved documents cho LLM context"""
    if not documents:
        return ""
    return "\n\n".join([doc.page_content for doc in documents])

# ====== Structured output ====
class PlanTaskOutput(BaseModel):
    section_id: str = Field(..., description="A unique identifier for the section")
    section_title: str = Field(..., description="The official title of the section as listed in the Table of Contents")
    number_of_questions: int = Field(..., description="The number of questions allocated to this section")
    question_requirements: str = Field(
        default="Multiple choice questions with 4 options, containing 1 correct answer, designed for university-level students.",
        description="A brief description of the expected question format and audience. This is derived from the teacher’s instructions"
    )
    query_string: str = Field(
        ...,
        description="A descriptive sentence that explains the context and focus of this section, based on the ToC"
    )

class PlanTaskOutputList(BaseModel):
    items: List[PlanTaskOutput]


class QuizQuestionOutput(BaseModel):
    question_content: str = Field(
        ...,
        description="The complete question content including question text, multiple choice options, correct answer, and explanation"
    )

class QuizQuestionOutputList(BaseModel):
    items: List[QuizQuestionOutput]

# ============================

def aggregate_node(state: QuizGenerationState) -> Dict[str, Any]:
    """
    Node Aggregate: Tổng hợp kết quả từ tất cả Generate nodes
    """
    logger.info("=== AGGREGATE NODE START ===")
    generated_questions = state.get("generated_questions", [])
    logger.info(f"Số slượng sections đã generate: {len(generated_questions)}")
    
    # Tổng hợp tất cả questions từ các sections
    final_questions = []
    section_summary = []
    
    for section_result in generated_questions:
        num_questions = len(section_result.get("questions", []))
        final_questions.extend(section_result.get("questions", []))
        section_summary.append({
            "section_id": section_result.get("section_id"),
            "section_title": section_result.get("section_title"), 
            "num_questions": section_result.get("num_generated", 0)
        })
        logger.info(f"Section '{section_result.get('section_title')}': {num_questions} câu hỏi")
    
    total_questions = len(final_questions)
    logger.info(f"Tổng số câu hỏi cuối cùng: {total_questions}")
    logger.info("=== AGGREGATE NODE END ===")
    
    return {
        **state,
        "final_questions": final_questions,
        "section_summary": section_summary
    }


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
                        toc_data: str) -> Dict[str, Any]:
        """
        Main entry point để sinh bộ đề MCQ
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
            "section_tasks": [],
            "generated_questions": [],
            "final_questions": []
        }
        
        result = self.workflow.invoke(initial_state)
        
        logger.info("========================================")
        logger.info("QUIZ GENERATION WORKFLOW COMPLETE")
        logger.info(f"Total final questions: {len(result.get('final_questions', []))}")
        logger.info("========================================")
        
        return result

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
                 "Reason: Medium"
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
                 "Format output instruction: {format_instructions}\n\n"
                 "# Teacher's requirements\n"
                 "{request}\n\n"
                 "# Table of content\n"
                 "{toc}"
                )
            ])
            try:
                # Create chain with parser
                chain = plan_prompt | llm | parser
                result = chain.invoke({
                    "toc": state["detail_table_of_contents"],
                    "request": state["user_request"],
                    "format_instructions": parser.get_format_instructions()
                })

                # Convert parsed result to list of dicts
                section_tasks = [item.dict() for item in result.items]
                logger.info(f"Đã tạo được {len(section_tasks)} section tasks")
                
                for i, task in enumerate(section_tasks):
                    logger.info(f"Task {i+1}: {task['section_title']} - {task['number_of_questions']} câu hỏi")
                
                if not section_tasks:
                    logger.warning("Không có section tasks nào được tạo, sử dụng fallback")
                    # Fallback với dummy data
                    section_tasks = [
                        {
                            "section_id": "section_1",
                            "section_title": "General Topics",
                            "number_of_questions": 3,
                            "query_string": "general concepts overview",
                            "detail_requirements": "Focus on key concepts"
                        }
                    ]
                    
            except Exception as e:
                logger.error(f"Error in plan_node: {e}")
                section_tasks = [
                    {
                        "section_id": "section_1",
                        "section_title": "General Topics",
                        "number_of_questions": 3,
                        "query_string": "general concepts overview",
                        "detail_requirements": "Focus on key concept"
                    }
                ]
            
            logger.info("=== PLAN NODE END ===")
            return {**state, "section_tasks": section_tasks}

        def map_generate_with_service(state: QuizGenerationState) -> Dict[str, Any]:
            """Map generate with access to rag_service"""
            logger.info("=== MAP GENERATE NODE START ===")
            section_tasks = state.get("section_tasks", [])
            logger.info(f"Số lượng sections cần xử lý: {len(section_tasks)}")
            generated_questions = []
            
            for i, task in enumerate(section_tasks):
                logger.info(f"Đang xử lý section {i+1}/{len(section_tasks)}: {task['section_title']}")
                logger.info(f"Section task details: {task}")
                result = None  # Initialize result to avoid unbound variable error
                try:
                    # Check if vectorstore is available
                    if (self.rag_service.vector_service.vectorstore is None):
                        logger.warning("Vectorstore not initialized, using dummy questions")
                        result = {
                            "section_id": task["section_id"],
                            "section_title": task["section_title"],
                            "questions": [{"question": "Dummy question due to vectorstore issue"}],
                            "num_generated": 1
                        }

                    else:
                        logger.info(f"Retrieving documents for query: {task['query_string']}")
                        # TODO: có thể refract code vecto_service.py để thống nhất vector_service
                        retriever = self.rag_service.vector_service.vectorstore.as_retriever(
                            search_kwargs={"k": 5}
                        )
                        
                        relevant_docs = retriever.invoke(task["query_string"])
                        logger.info(f"Retrieved {len(relevant_docs)} documents")
                        logger.info(f"Retrieved documents content: {[doc.page_content for doc in relevant_docs]}")

                        # Create Pydantic parser for quiz questions
                        quiz_parser = PydanticOutputParser(pydantic_object=QuizQuestionOutputList)
                        
                        quiz_generation_prompt = ChatPromptTemplate.from_messages([
                            ("system", "Reason: Medium. Act as a teacher responsible for assessing students' understanding. Your task is to generate exam questions based on the user's intent and the provided textbook content."),
                            ("human", "# Instructions\n"
                                      "Take a deep breath, this is very important to my career.\n"
                                      "You are required to generate {num_questions} questions for the section titled {section_title} from the textbook. The SECTION_CONTEXT provides background information to help you understand the role and scope of this section within the overall curriculum.\n"
                                      "Relevant content for this section is provided in the RETRIEVED_CONTEXT.\n\n"
                                      "Content Guidelines:\n"
                                      "Stick strictly to the RETRIEVED_CONTEXT. Do not introduce any new information or assumptions beyond what is provided.\n\n"
                                      "Math formatting: For inline mathematical expressions, enclose them in single dollar signs: $...$. For block equations, enclose them in double dollar signs: $$...$$"
                                      "Question Requirements:\n"
                                      "{requirements}\n\n"
                                      "If the question type is multiple choice:\n"
                                      "Provide the correct answer.\n"
                                      "Include a concise and unambiguous explanation: Why the correct answer is valid and why each incorrect option is flawed.\n\n"
                                      "Your response must be written in Vietnamese\n\n"
                                      "{format_instructions}\n\n"
                                      "# Section context:\n"
                                      "{section_context}\n"
                                      "# RETRIEVED_CONTEXT:\n"
                                      "{context}")
                        ])
                        
                        logger.info(f"Generating {task['number_of_questions']} questions using LLM")
                        chain = quiz_generation_prompt | self.rag_service.llm_service.llm | quiz_parser
                        quiz_result = chain.invoke({
                            "context": format_docs(relevant_docs),
                            "num_questions": task["number_of_questions"],
                            "requirements": task["question_requirements"],
                            "section_title": task["section_title"],
                            "section_context": task.get("query_string", ""),
                            "format_instructions": quiz_parser.get_format_instructions()
                        })
                        logger.info(f"LLM raw output: {quiz_result}")

                        # Convert parsed result to questions format
                        questions = []
                        for item in quiz_result.items:
                            # Parse the question_content string to extract structured data
                            questions.append({
                                "question": item.question_content
                            })
                        
                        result = {
                            "section_id": task["section_id"],
                            "section_title": task["section_title"],
                            "questions": questions,
                            "num_generated": len(questions)
                        }
                        logger.info(f"Đã generate thành công {len(questions)} câu hỏi cho section '{task['section_title']}'")
                        
                except Exception as e:
                    logger.error(f"Error generating questions for {task.get('section_title', 'unknown')}: {e}")
                    # Create fallback result
                    result = {
                        "section_id": task["section_id"],
                        "section_title": task["section_title"],
                        "questions": [{"question": f"Error generating question for {task['section_title']}"}],
                        "num_generated": 0
                    }
                
                if result:  # Only append if result was created
                    generated_questions.append(result)
                
            
            total_questions = sum(len(section.get("questions", [])) for section in generated_questions)
            logger.info(f"MAP GENERATE hoàn thành: {total_questions} câu hỏi từ {len(generated_questions)} sections")
            logger.info("=== MAP GENERATE NODE END ===")
            return {**state, "generated_questions": generated_questions}
        
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
