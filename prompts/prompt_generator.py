from langchain_core.prompts import PromptTemplate
template=PromptTemplate(
   template= """
 hii please get the name is {name_query} and tell about his date of birth and his role

""",
input_variables=['name_query'],
validate_template=True
)

template.save('./prompts/template.json')