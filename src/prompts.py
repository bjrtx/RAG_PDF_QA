PROMPTS = {
    "RAG_PDF": (
        "I want you to answer a question based on a chunk of a retrieved file "
        "that I will give you. If you don't find the answer in the text that"
        " I give you, answer: 'I don't find anything in the corresponding "
        "text'. First write the page number from the PDF:{page_number} if"
        "there is one, then answer the question: {question} with the "
        "text: {content}."
    ),
    "RAG_PDFs_data": (
        "I want you to answer a question based on information retrieved across "
        "multiple PDFs. If you don't find the answer in the texts that I give "
        "you, answer: 'I don't find anything in the corresponding texts'. "
        "First, write the source file name:{filename} and the page number "
        "from the PDF:{page_number}, then answer the question: {question} "
        "using the relevant text: {content}."
    ),
}
