import random

# create demo llm class
class NakliLLm:

    def __init__(self):
        print('llm created')

    def predict(self,prompt):
        response_list=[
            'delhi is the india capital',
            'deep learning is the ai field',
            'next js is the react libruary'
        ]

        return {'response':random.choice(response_list)}
    
# create a template class

class NakliPromptTemplate:
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def format(self,input_dict):
        return self.template.format(**input_dict)
    

# create chain

class NakliLLmChain:
    def __init__(self,prompt,llm):
        self.prompt=prompt
        self.llm=llm

    def run(self,input_dict):
       final_prompt= self.prompt.format(input_dict)   
       response=self.llm.predict(final_prompt)
       return response['response']         


template=NakliPromptTemplate(
    template='write a poem {topic}',
    input_variables=['topic']
)

llm=NakliLLm()


prompt =template.format({'topic':'deep learing'})

chain=NakliLLmChain(prompt,llm) 
result=chain.run({'topic':'deep learning'})

print(result)
        

    
