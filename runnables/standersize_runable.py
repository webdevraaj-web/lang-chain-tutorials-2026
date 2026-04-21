from abc import ABC ,abstractmethod
import random



# create runable class

class Runable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


# create random llm class
class NakliLLm(Runable):

    def __init__(self):
        print('llm created')
    
    def invoke(self,prompt):
        response_list=[
            'delhi is the india capital',
            'deep learning is the ai field',
            'next js is the react libruary'
        ]

        return {'response':random.choice(response_list)}

    def predict(self,prompt):
        response_list=[
            'delhi is the india capital',
            'deep learning is the ai field',
            'next js is the react libruary'
        ]

        return {'response':random.choice(response_list)}
    

# create template


class NakliPromptTemplate(Runable):
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def invoke(self,input_dict):
         return self.template.format(**input_dict)  

    def format(self,input_dict):
        return self.template.format(**input_dict)    
    


# create runnable connector to join the chains togather

class RunableConnector(Runable):
    def __init__(self,runable_list):

        self.runable_list=runable_list

    def invoke(self,input_data):
        for runable in self.runable_list:
            input_data=runable.invoke(input_data)
        return input_data
    
# create strparser

class NakliStrOutputParser(Runable):
    def __init__(self):
        pass

    def invoke(self,input_data):
        return input_data['response']


llm=NakliLLm()
parser=NakliStrOutputParser()
prompt1=NakliPromptTemplate(
    template='write a poem {topic}',
    input_variables=['topic']
)

prompt2=NakliPromptTemplate(
    template='write a poem {response}',
    input_variables=['response']
)

chain1=RunableConnector([prompt1,llm])
chain2=RunableConnector([prompt2,llm,parser])

final_chain=RunableConnector([chain1,chain2])

result=final_chain.invoke({'topic':'data'})

print(result)

 