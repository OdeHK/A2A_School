from langchain.prompts import ChatPromptTemplate

def get_plan_prompt() -> ChatPromptTemplate:
    """
    Prompt template for planning quiz generation tasks.
    
    Returns:
        ChatPromptTemplate: Prompt for creating question distribution plan
    """
    return ChatPromptTemplate.from_messages([
        ("system", 
         "Reasoning: Medium"
         "Your task is to design a question distribution plan for an exam set. "
         "You are not generating the actual questions, only planning how the knowledge should be allocated across the test."
        ),
        ("human", 
         "Input: You will be provided with:\n"
         "The Table of Contents from the textbook or curriculum, organized hierarchically.\n"
         "The teacher's requirements regarding exam content, such as preferred question types, target audience, or emphasis on specific topics.\n"
         "Output: Return a list of tasks in the form of a JSON object. Each task corresponds to a lowest-level section (leaf node) from the Table of Contents. "
         "The total number of questions across all tasks must match the overall exam question count.\n"
         "List at most 3 tasks. Select the 3 most important sections. Do not include any task with number_of_questions = 0."
         "Each task must include the following fields:\n"
         "section_id (string): A unique identifier for the section\n"
         "section_title (string): The official title of the section as listed in the Table of Contents\n"
         "number_of_questions (positive int): The number of questions allocated to this section. This should reflect the importance or emphasis based on teacher input and curriculum weight.\n"
         "question_requirements (string): A brief description of the expected question format and audience. This is derived from the teacher's instructions. Default (if unspecified): \"Multiple choice questions with 4 options, containing 1 correct answer, designed for university-level students.\"\n"
         "query_string (string): Using the section title and its immediate parent section title from the Table of Contents, write one descriptive sentence that explains the context and focus of this section. The sentence should reflect the hierarchical structure of the curriculum and highlight key concepts or themes relevant to the section. The sentence must be written in the same language used in the Table of Contents\n\n"
         "Format output instruction:\n {format_instructions}\n\n"
         "# Teacher's requirements\n"
         "{request}\n\n"
         "# Table of content\n"
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