from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage

def get_plan_prompt() -> ChatPromptTemplate:
    """
    Prompt template for planning quiz generation tasks.
    
    Returns:
        ChatPromptTemplate: Prompt for creating question distribution plan
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
         "Reasoning: medium. "
         "# Instructions: You are a curriculum planner tasked with creating a question distribution plan for an exam based on the provided Table of Contents and teacher's requirements. Please carefully follow this step-by-step process to ensure accuracy and completeness in your planning:\n\n"
         "1. Based on the teacher's requirements, identify the lowest-level sections (leaf nodes) in the Table of Contents that are truly relevant to the requested knowledge scope.\n"
         "2. The maximum number of sections to select is 15. Therefore, prioritize the most important sections.\n"
         "3. If the teacher specifies a preference for certain topics or sections, allocate more questions to those areas accordingly. If no preferences are given, distribute questions evenly across all sections. The total number of questions across all selected sections must match the overall exam question count.\n"
         "4. The maximum number of questions for each section is 20. Ensure that no section exceeds this limit. So that, same section can have multiple tasks if needed.\n"
         "5. Important note: Prefer allocating more questions to fewer sections rather than selecting many sections with few questions each. This is critical for cost efficiency.\n\n"
         "# Response format: {format_instructions}"
        ),
        HumanMessagePromptTemplate.from_template(
        "I will explain the information you receive and the expected output.\n\n"
        "# INPUT:\n"
        "- The Table of Contents from the textbook or curriculum, organized hierarchically.\n"
        "- The teacher's requirements regarding exam content, such as preferred question types, target audience, or emphasis on specific topics.\n\n"
        "# OUTPUT:\n"
        "Return a list of tasks in the form of a JSON object. Each task corresponds to a lowest-level section (leaf node) from the Table of Contents. Do not include sections with zero questions.\n\n"
        "Each task must include the following fields:\n"
        "- section_id (string): A unique identifier for the section\n"
        "- section_title (string): The official title of the section as listed in the Table of Contents\n"
        "- number_of_questions (positive int): The number of questions allocated to this section. This should reflect the importance or emphasis based on teacher input and curriculum weight.\n"
        "- question_requirements (string): A brief description of the expected question format and audience. This is derived from the teacher's instructions. Default (if unspecified): \"Multiple choice questions with 4 options, containing 1 correct answer, designed for university-level students.\"\n"
        "- query_string (string): Using the section title and its immediate parent section title from the Table of Contents, write one descriptive sentence that explains the context and focus of this section. The sentence should reflect the hierarchical structure of the curriculum and highlight key concepts or themes relevant to the section. The sentence must be written in the same language used in the Table of Contents.\n\n"
        "# Teacher's requirements:\n"
        "{request}\n\n"
        "# Table of Contents:\n"
        "{toc}\n\n"
        "# TASK:\n"
        "Following the instructions above, create a content distribution plan for the exam. Strictly adhere to the guidelines and output format requirements."
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
            "# Instructions: You are a curriculum planner tasked with creating a question distribution plan for an exam based on the provided Table of Contents and teacher's requirements. As the first agent in a multi-agent workflow, your responsibility is to decompose the overall task into smaller, well-defined sub-tasks for sub-agents to execute. Please carefully follow this step-by-step process to ensure accuracy and completeness in your planning:\n\n"
            "1. Based on the teacher's requirements, identify the lowest-level sections (leaf nodes) in the Table of Contents that are truly relevant to the requested knowledge scope. Each identified section will be delegated as an independent sub-task to specialized sub-agents in the next step.\n"
            "2. The maximum number of sections to select is 15. Therefore, prioritize the most important sections.\n"
            "3. If the teacher specifies a preference for certain topics or sections, allocate more questions to those areas accordingly. If no preferences are given, distribute questions evenly across all sections. The total number of questions across all selected sections must match the overall exam question count.\n"
            "4. The maximum number of questions allowed per section is 20; if this limit is exceeded, the section should be split across multiple tasks.\n"
            "5. Important note: Prefer allocating more questions to fewer sections rather than selecting many sections with few questions each. This is critical for cost efficiency.\n\n"
            "# Response format: {format_instructions}"
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
