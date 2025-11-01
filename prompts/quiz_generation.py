from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage

def get_plan_prompt() -> ChatPromptTemplate:
    """
    Prompt template for planning quiz generation tasks.
    
    Returns:
        ChatPromptTemplate: Prompt for creating question distribution plan
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(
         "Reasoning: Medium."
         "Your task is to design a question distribution plan for an exam set. You are not generating the actual questions, only planning how the knowledge should be allocated across the test."
        ),
        HumanMessagePromptTemplate.from_template( 
        "### ROLE: \n" \
        "You are a curriculum planner tasked with creating a question distribution plan for an exam based on the provided Table of Contents and teacher's requirements.\n\n"
        "### INPUT: \n" \
        "The Table of Contents from the textbook or curriculum, organized hierarchically.\n"
        "The teacher's requirements regarding exam content, such as preferred question types, target audience, or emphasis on specific topics.\n"
        "### OUTPUT: Return a list of tasks in the form of a JSON object. Each task corresponds to a lowest-level section (leaf node) from the Table of Contents. \n "
        "Each task must include the following fields:\n"
        "section_id (string): A unique identifier for the section\n"
        "section_title (string): The official title of the section as listed in the Table of Contents\n"
        "number_of_questions (positive int): The number of questions allocated to this section. This should reflect the importance or emphasis based on teacher input and curriculum weight.\n"
        "question_requirements (string): A brief description of the expected question format and audience. This is derived from the teacher's instructions. Default (if unspecified): \"Multiple choice questions with 4 options, containing 1 correct answer, designed for university-level students.\"\n"
        "query_string (string): Using the section title and its immediate parent section title from the Table of Contents, write one descriptive sentence that explains the context and focus of this section. The sentence should reflect the hierarchical structure of the curriculum and highlight key concepts or themes relevant to the section. The sentence must be written in the same language used in the Table of Contents\n\n"
        "### CONSTRAINTS:\n"
        "1. List at most 15 tasks, so prioritize the most important sections if there are too many sections. Do not include any task with number_of_questions = 0.\n"
        "2. The total number of questions across all tasks must match the overall exam question count.\n"
        "3. If the teacher specifies a preference for certain topics or sections, allocate more questions to those areas accordingly. If no preferences are given, distribute questions evenly across all sections.\n"
        "### Format output instruction:\n {format_instructions}\n\n"
        "### Teacher's requirements\n"
        "{request}\n\n"
        "### Table of content\n"
        "{toc}"
        )
    ])

def get_quiz_generation_prompt() -> ChatPromptTemplate:
    """
    Prompt template for generating quiz questions from content.
    
    Returns:
        ChatPromptTemplate: Prompt for generating quiz questions based on retrieved context
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Reasoning: medium. "
            "# Instructions: Act as an instructor responsible for evaluating students' comprehension. Your task is to generate exam questions based on the user's intent and the provided textbook content."
            "The scope of knowledge must strictly revolve around the provided material. Each question must be fully contextualized, as students are assumed to have no prior knowledge of the given content. The difficulty level should align appropriately with the students' academic level—neither overly simplistic nor excessively advanced. "
            "The number of questions generated must exactly match the requested quantity. All outputs must be produced in Vietnamese. "
            "# Response format: \n Inline LaTeX with $ for mathematical expressions is permitted. {format_instructions}. It's not necessary for wrapping your response in ```json``` and use escape character in JSON."
        ),
        HumanMessagePromptTemplate.from_template(
            "This is an explanation of the information I provide: \n"
            "Question Requirements: Specific guidelines on the types of questions to generate, including format, difficulty level, target audience, and any special instructions.\n"
            "Section Title: The title of the textbook section from which the quiz is to be derived.\n"
            "Section Context: Background information about the section to clarify its role and scope within the overall textbook.\n"
            "Retrieved Context: Relevant content extracted from the textbook pertaining to the section.\n\n"
            "# INPUT DETAILS:\n"
            "Number of Questions: {num_questions}\n"
            "Question Requirements: {requirements}\n"
            "Section Title: {section_title}\n"
            "Section Context: {section_context}\n\n"
            "Retrieved Context: {context}\n\n"
            "# TASK: "
            "Generate a set of questions (in Vietnamese) in accordance with the above instructions. Note that strict adherence to the specified output formatting guidelines is absolutely critical and non-negotiable."
        )])