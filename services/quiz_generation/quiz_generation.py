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
    username: str  
    detail_table_of_contents: str
    user_request: str
    section_tasks: PlanTaskOutputList  # Tasks cho map-reduce
    generated_questions: QuizQuestionOutput  # Kết quả từ từng Generate node
    final_questions: List[Dict[str, Any]]  # Kết quả cuối cùng
    type_requirements: Dict[str, int]  # Question type requirements: {"essay": 5, "multiple_choice": 6}


class QuizGenerationService:
    """Main service điều phối việc sinh Quiz sử dụng LangGraph"""

    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        self.llm_service = rag_service.llm_service
        self.vector_service = rag_service.vector_service
        self.workflow = self._create_workflow()

    def generate_quiz_set(self, 
                        document_id: str,
                        username: str,
                        user_request: str,
                        toc_data: str) -> str:
        """
        Main entry point để sinh bộ đề MCQ
        Args:
            document_id: ID của tài liệu tham khảo
            username: Username for metadata filtering
            user_request: Yêu cầu của giáo viên
            toc_data: Dữ liệu mục lục chi tiết của tài liệu
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
            "final_questions": [],
            "type_requirements": {}  # Will be populated in plan_node
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
                total_planned_questions = 0
                for i, task in enumerate(section_tasks.tasks):
                    logger.info(f"Task {i+1}: {task.section_title} - {task.number_of_questions} câu hỏi")
                    total_planned_questions += task.number_of_questions
                
                logger.info(f"📊 Total planned questions FROM LLM: {total_planned_questions}")
                logger.info(f"📊 User request: {state['user_request']}")
                
                # Extract expected total AND question type requirements from user request
                import re
                user_request = state["user_request"].lower()
                
                # Parse type-specific requirements with improved pattern matching
                # Strategy: Find all number + type pairs in the request
                type_requirements = {}
                
                # Combined pattern to capture: number + optional "câu" + question type
                # This will match patterns like:
                # - "5 câu tự luận" or "5 tự luận"
                # - "6 câu trắc nghiệm" or "6 trắc nghiệm"
                # - "3 essay" or "3 câu essay"
                
                # Pattern explanation:
                # (\d+) - captures the number
                # \s* - optional whitespace
                # (?:câu\s+)? - optional "câu" word with whitespace
                # (tự\s*luận|essay|essai|trắc\s*nghiệm|multiple[\s-]?choice|mc|đúng\s*sai|true[\s-]?false) - question type
                
                pattern = r'(\d+)\s*(?:câu\s+)?(tự\s*luận|essay|essai|trắc\s*nghiệm|multiple[\s-]?choice|mc|đúng\s*sai|true[\s-]?false)'
                matches = re.findall(pattern, user_request)
                
                logger.info(f"🔍 Regex matches found: {matches}")
                
                for number, question_type in matches:
                    count = int(number)
                    # Normalize question type to standard names
                    if re.search(r'tự\s*luận|essay|essai', question_type):
                        type_requirements['essay'] = type_requirements.get('essay', 0) + count
                        logger.info(f"✅ Found essay requirement: +{count} (total: {type_requirements['essay']})")
                    elif re.search(r'trắc\s*nghiệm|multiple[\s-]?choice|mc', question_type):
                        type_requirements['multiple_choice'] = type_requirements.get('multiple_choice', 0) + count
                        logger.info(f"✅ Found multiple choice requirement: +{count} (total: {type_requirements['multiple_choice']})")
                    elif re.search(r'đúng\s*sai|true[\s-]?false', question_type):
                        type_requirements['true_false'] = type_requirements.get('true_false', 0) + count
                        logger.info(f"✅ Found true/false requirement: +{count} (total: {type_requirements['true_false']})")
                
                logger.info(f"📊 Final type requirements: {type_requirements}")
                logger.info(f"📊 Expected total from types: {sum(type_requirements.values())} questions")
                
                # Calculate expected total from type requirements or general pattern
                if type_requirements:
                    expected_total = sum(type_requirements.values())
                    logger.info(f"Expected total from type requirements: {expected_total} ({type_requirements})")
                else:
                    # Fallback: Try to find general number like "10 câu" or "10 câu hỏi"
                    numbers = re.findall(r'(\d+)\s*câu', user_request)
                    if numbers:
                        expected_total = sum(int(n) for n in numbers)
                        logger.info(f"Expected total from general pattern: {expected_total}")
                    else:
                        expected_total = 0
                
                # Store type requirements in state for aggregate_node validation
                state["type_requirements"] = type_requirements
                
                # If total planned exceeds expected, proportionally scale down
                if expected_total > 0 and total_planned_questions > expected_total:
                    logger.warning(f"⚠️ LLM OVERPLANNED: Planned {total_planned_questions} but expected {expected_total}. Scaling down...")
                    scale_factor = expected_total / total_planned_questions
                    
                    # Scale and round, ensuring at least 1 question per task
                    scaled_tasks = []
                    remaining = expected_total
                    for task in section_tasks.tasks[:-1]:  # All but last
                        scaled_num = max(1, round(task.number_of_questions * scale_factor))
                        scaled_num = min(scaled_num, remaining - (len(section_tasks.tasks) - len(scaled_tasks) - 1))
                        task.number_of_questions = scaled_num
                        scaled_tasks.append(task)
                        remaining -= scaled_num
                    
                    # Last task gets whatever remains
                    if section_tasks.tasks:
                        section_tasks.tasks[-1].number_of_questions = max(1, remaining)
                        scaled_tasks.append(section_tasks.tasks[-1])
                    
                    section_tasks.tasks = scaled_tasks
                    logger.info(f"✅ Scaled tasks to match expected total of {expected_total}")
                    
                    # Recalculate total after scaling
                    total_planned_questions = sum(task.number_of_questions for task in section_tasks.tasks)
                    logger.info(f"📊 Total after scaling: {total_planned_questions}")
                    
                    # Log adjusted plan
                    for i, task in enumerate(section_tasks.tasks):
                        logger.info(f"Adjusted Task {i+1}: {task.section_title} - {task.number_of_questions} câu hỏi")
                
                # Critical check: If LLM underplanned - FORCE FIX!
                elif expected_total > 0 and total_planned_questions < expected_total:
                    logger.error(f"❌ LLM UNDERPLANNED: Planned {total_planned_questions} but expected {expected_total}!")
                    missing = expected_total - total_planned_questions
                    logger.error(f"❌ Missing {missing} questions!")
                    logger.warning(f"🔧 APPLYING FORCE FIX: Adding missing questions to tasks...")
                    
                    if section_tasks.tasks:
                        # Strategy: Distribute missing questions across tasks evenly
                        # Add to first task (or distribute round-robin)
                        tasks_count = len(section_tasks.tasks)
                        
                        # Simple approach: Add all missing to first task
                        section_tasks.tasks[0].number_of_questions += missing
                        logger.info(f"✅ Added {missing} questions to Task 1 ({section_tasks.tasks[0].section_title})")
                        logger.info(f"✅ Task 1 new total: {section_tasks.tasks[0].number_of_questions} questions")
                        
                        # Recalculate
                        total_planned_questions = sum(task.number_of_questions for task in section_tasks.tasks)
                        logger.info(f"📊 Total after force fix: {total_planned_questions}")
                        
                        # Verify
                        if total_planned_questions == expected_total:
                            logger.info(f"✅✅✅ PERFECT! Now total = expected = {expected_total}")
                        else:
                            logger.error(f"❌ Still mismatch after fix: {total_planned_questions} != {expected_total}")
                    else:
                        logger.error(f"❌ Cannot fix - no tasks available!")
                
                # If we have type requirements, distribute them across tasks
                if type_requirements:
                    total_tasks = len(section_tasks.tasks)
                    if total_tasks > 0:
                        logger.info("=== TYPE DISTRIBUTION ALGORITHM ===")
                        logger.info(f"Type requirements: {type_requirements}")
                        logger.info(f"Total tasks: {total_tasks}")
                        
                        # STRATEGY: Greedy allocation - assign types to tasks sequentially
                        # This avoids rounding errors from proportional distribution
                        
                        # Create a pool of questions to distribute
                        question_pool = []
                        for q_type, q_count in type_requirements.items():
                            for _ in range(q_count):
                                question_pool.append(q_type)
                        
                        logger.info(f"Question pool to distribute: {question_pool}")
                        
                        # Distribute questions to tasks round-robin style
                        task_allocations = [[] for _ in range(total_tasks)]
                        for idx, q_type in enumerate(question_pool):
                            task_idx = idx % total_tasks  # Round-robin assignment
                            task_allocations[task_idx].append(q_type)
                        
                        # Update each task's requirements
                        for i, task in enumerate(section_tasks.tasks):
                            allocated = task_allocations[i]
                            if allocated:
                                # Count each type
                                type_counts = {}
                                for q_type in allocated:
                                    type_counts[q_type] = type_counts.get(q_type, 0) + 1
                                
                                # Build requirement string
                                task_types = []
                                for q_type, count in type_counts.items():
                                    if q_type == 'essay':
                                        task_types.append(f"{count} câu tự luận")
                                    elif q_type == 'multiple_choice':
                                        task_types.append(f"{count} câu trắc nghiệm")
                                
                                type_spec = " và ".join(task_types)
                                
                                # Update task number_of_questions to match allocation
                                task.number_of_questions = len(allocated)
                                
                                # Append to existing requirements
                                if task.question_requirements:
                                    task.question_requirements += f". Include: {type_spec}"
                                else:
                                    task.question_requirements = f"Include: {type_spec}"
                                
                                logger.info(f"Task {i+1} ({task.section_title}): {len(allocated)} questions - {type_spec}")
                            else:
                                logger.info(f"Task {i+1} ({task.section_title}): No questions allocated")
                        
                                # Verify total allocation
                        total_allocated = sum(len(alloc) for alloc in task_allocations)
                        logger.info(f"✅ Total allocated: {total_allocated}, Expected: {expected_total}")
                        
                        # Critical check: Verify allocation matches requirements
                        allocated_by_type = {}
                        for alloc in task_allocations:
                            for q_type in alloc:
                                allocated_by_type[q_type] = allocated_by_type.get(q_type, 0) + 1
                        
                        logger.info(f"✅ Allocated by type: {allocated_by_type}")
                        logger.info(f"📌 Required by type: {type_requirements}")
                        
                        # Check for mismatches
                        if allocated_by_type != type_requirements:
                            logger.error(f"❌ MISMATCH! Allocated {allocated_by_type} != Required {type_requirements}")
                        else:
                            logger.info(f"✅ Perfect match! Allocated == Required")
                        
                        logger.info("=== END TYPE DISTRIBUTION ===")
                
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
            """Map generate with access to rag_service and metadata filtering"""
            logger.info("=== MAP GENERATE NODE START ===")
            section_tasks = state.get("section_tasks", PlanTaskOutputList(tasks=[]))
            document_id = state.get("document_id")
            username = state.get("username")
            
            logger.info(f"Số lượng sections cần xử lý: {len(section_tasks.tasks)}")
            logger.info(f"Document ID: {document_id}, Username: {username}")
            
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
                        
                        # Prepare metadata filter for document-specific and user-specific queries
                        metadata_filter = {
                            "$and": [
                                {"document_id": document_id},
                                {"username": username}
                            ]
                        }
                        logger.info(f"Applying metadata filter: {metadata_filter}")
                        
                        # Use retrieve_documents with metadata filtering
                        relevant_docs = self.rag_service.retrieve_documents(
                            query=task.query_string,
                            top_k=5,
                            filter=metadata_filter
                        )
                        
                        logger.info(f"Retrieved {len(relevant_docs)} documents")
                        logger.info(f"Retrieved documents content: {[doc.page_content for doc in relevant_docs]}")

                        # Create Pydantic parser for quiz questions
                        quiz_parser = PydanticOutputParser(pydantic_object=QuizQuestionOutput)
                        
                        quiz_generation_prompt = ChatPromptTemplate.from_messages([
                            ("system", 
                             "Reasoning: High - Use reverse reasoning approach\n\n"
                             "You are an expert teacher creating exam questions. Use this REVERSE REASONING process:\n\n"
                             "🔄 REVERSE REASONING PROCESS:\n"
                             "STEP 1 (GOAL): Read 'Question Requirements' first - What EXACTLY must I generate?\n"
                             "  - Look for: 'X câu tự luận' → GOAL: X essay questions\n"
                             "  - Look for: 'Y câu trắc nghiệm' → GOAL: Y multiple_choice questions\n"
                             "  - Total: {num_questions} questions\n\n"
                             "STEP 2 (PLAN): Create a mental checklist BEFORE writing questions:\n"
                             "  - [ ] Write ___ essay questions (type='essay')\n"
                             "  - [ ] Write ___ multiple_choice questions (type='multiple_choice')\n"
                             "  - Total slots: {num_questions}\n\n"
                             "STEP 3 (EXECUTE): Generate questions following your checklist EXACTLY\n"
                             "  - Generate essay questions FIRST if required\n"
                             "  - Then generate multiple_choice questions if required\n\n"
                             "STEP 4 (VERIFY): Count backwards from output to requirement:\n"
                             "  - Count questions with type='essay' → Does it match requirement?\n"
                             "  - Count questions with type='multiple_choice' → Does it match requirement?\n"
                             "  - If mismatch: STOP and regenerate correctly\n\n"
                             "⚠️ CRITICAL MAPPING:\n"
                             "  'tự luận' = 'essay' → type='essay'\n"
                             "  'trắc nghiệm' = 'multiple choice' → type='multiple_choice'\n\n"
                             "❌ COMMON MISTAKE TO AVOID:\n"
                             "  If requirement says '1 câu tự luận và 3 câu trắc nghiệm'\n"
                             "  WRONG: Creating 3 essay + 1 multiple_choice (reversed!)\n"
                             "  RIGHT: Creating 1 essay + 3 multiple_choice (as stated!)\n\n"
                             "Remember: 'tự luận' comes FIRST in Vietnamese but must map to type='essay' in JSON!\n"
                            ),
                            ("human", 
                             "Task: Generate {num_questions} questions for section '{section_title}'\n\n"
                             "📋 Question Requirements (READ THIS CAREFULLY!):\n"
                             "{requirements}\n\n"
                             "⚠️ VERIFICATION CHECKLIST (Do this before submitting):\n"
                             "□ Count essay questions (type='essay') - does it match the requirement?\n"
                             "□ Count multiple_choice questions (type='multiple_choice') - does it match the requirement?\n"
                             "□ Total questions = {num_questions}?\n"
                             "□ All content based on RETRIEVED_CONTEXT below?\n\n"
                             "SECTION_CONTEXT:\n"
                             "{section_context}\n\n"
                             "RETRIEVED_CONTEXT:\n"
                             "{context}\n\n"
                             "Response Format:\n{format_instructions}\n\n"
                             "Math formatting: For inline expressions use $...$, for block equations use $$...$$\n"
                             "Response must be in Vietnamese\n\n"
                             "⚠️ FINAL REMINDER: Check question TYPES match requirements EXACTLY before responding!"
                            )
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
                    
                
            logger.info(f"MAP GENERATE hoàn thành: {len(generated_questions.questions)} câu hỏi")
            logger.info("=== MAP GENERATE NODE END ===")
            return {**state, "generated_questions": generated_questions}
        
        def aggregate_node(state: QuizGenerationState) -> Dict[str, Any]:
            """
            Node Aggregate: Tổng hợp kết quả từ tất cả Generate nodes
            Validates and trims questions by type if type_requirements are specified
            """
            logger.info("=== AGGREGATE NODE START ===")
            generated_questions = state.get("generated_questions", QuizQuestionOutput(questions=[]))
            logger.info(f"Số lượng questions đã generate: {len(generated_questions.questions)}")
            
            if generated_questions.questions:
                # Check if we have type requirements to validate
                type_requirements = state.get("type_requirements", {})
                
                if type_requirements:
                    logger.info(f"Type requirements found: {type_requirements}")
                    
                    # Count generated questions by type
                    type_counts = {}
                    questions_by_type = {}
                    for q in generated_questions.questions:
                        q_type = q.type
                        type_counts[q_type] = type_counts.get(q_type, 0) + 1
                        if q_type not in questions_by_type:
                            questions_by_type[q_type] = []
                        questions_by_type[q_type].append(q)
                    
                    logger.info(f"Generated questions by type: {type_counts}")
                    logger.info(f"Required questions by type: {type_requirements}")
                    
                    # Check if we have the exact types required
                    validated_questions = []
                    missing_info = []
                    
                    for req_type, req_count in type_requirements.items():
                        available = questions_by_type.get(req_type, [])
                        actual_count = len(available)
                        
                        if actual_count >= req_count:
                            # Take exactly the required number
                            validated_questions.extend(available[:req_count])
                            logger.info(f"✓ {req_type}: Using {req_count} out of {actual_count} available")
                        elif actual_count > 0:
                            # Use all available (fewer than required)
                            validated_questions.extend(available)
                            missing_info.append(f"{req_type}: need {req_count}, got {actual_count}")
                            logger.warning(f"⚠ {req_type}: Insufficient questions - need {req_count}, got {actual_count}")
                        else:
                            # No questions of this type generated
                            missing_info.append(f"{req_type}: need {req_count}, got 0")
                            logger.error(f"✗ {req_type}: NO questions generated - needed {req_count}")
                    
                    # Update generated_questions with validated list
                    generated_questions = QuizQuestionOutput(questions=validated_questions)
                    logger.info(f"Final validated question count: {len(validated_questions)}")
                    
                    # Log final distribution
                    final_type_counts = {}
                    for q in validated_questions:
                        final_type_counts[q.type] = final_type_counts.get(q.type, 0) + 1
                    logger.info(f"Final questions by type: {final_type_counts}")
                    
                    # If there are missing questions, log a summary
                    if missing_info:
                        logger.warning(f"Question type mismatches: {'; '.join(missing_info)}")
                        logger.warning("Consider adjusting the prompt or regenerating questions")
                
                # Convert to human-readable list
                final_questions = QuizGenerationService._convert_quiz_question_output_to_list(questions=generated_questions)
            
                # Write to file for record-keeping
                QuizGenerationService._write_questions_to_file(questions=generated_questions)
                logger.info("Written generated questions to file")
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
    def _write_questions_to_file(questions: QuizQuestionOutput):
        """Write generated questions to a JSON file for record-keeping"""
        # TODO: Đây cách tiếp cận tạm thời, cần cải thiện sau
        file_path = "session_data\\temp\\quiz_data.json" 

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(questions.model_dump(), f, ensure_ascii=False, indent=4)
        logger.info(f"Generated questions written to {file_path}")
