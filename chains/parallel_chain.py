from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()


llm1=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation"
)

llm2=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm1)
model2=ChatHuggingFace(llm=llm2)


template1=PromptTemplate(
    template='generate short and simple notes on the following text \n {text}',
    input_variables=['text']
)

template2=PromptTemplate(
    template='generate five question and answer on the following \n {text}',
    input_variables=['text']
)

template3=PromptTemplate(
    template='merge both provided notes and faqs in the single document notes-> \n {notes} or \n {faqs} ',
    input_variables=['notes','faqs']
)



parser=StrOutputParser()


parallel_chain=RunnableParallel({
    'notes': template1 | model1 | parser,
    'faqs': template2 | model2 | parser
})

merge_chain= template3 | model1 | parser


chain=parallel_chain | merge_chain

text='''

Importantly, a deep learning process can learn which features to optimally place at which level on its own. Prior to deep learning, machine learning techniques often involved hand-crafted feature engineering to transform the data into a more suitable representation for a classification algorithm to operate on. In the deep learning approach, features are not hand-crafted and the model discovers useful feature representations from the data automatically. This does not eliminate the need for hand-tuning; for example, varying numbers of layers and layer sizes can provide different degrees of abstraction.[8][2]

The word "deep" in "deep learning" refers to the number of layers through which the data is transformed. More precisely, deep learning systems have a substantial credit assignment path (CAP) depth. The CAP is the chain of transformations from input to output. CAPs describe potentially causal connections between input and output. For a feedforward neural network, the depth of the CAPs is that of the network and is the number of hidden layers plus one (as the output layer is also parameterized). For recurrent neural networks, in which a signal may propagate through a layer more than once, the CAP depth is potentially unlimited.[9] No universally agreed-upon threshold of depth divides shallow learning from deep learning, but most researchers agree that deep learning involves CAP depth higher than two. CAP of depth two has been shown to be a universal approximator in the sense that it can emulate any function.[10] Beyond that, more layers do not add to the function approximator ability of the network. Deep models (CAP > two) are able to extract better features than shallow models and hence, extra layers help in learning the features effectively.

Deep learning architectures can be constructed with a greedy layer-by-layer method.[11] Deep learning helps to disentangle these abstractions and pick out which features improve performance.[8]

Deep learning algorithms can be applied to unsupervised learning tasks. This is an important benefit because unlabeled data is more abundant than labeled data. Examples of deep structures that can be trained in an unsupervised manner are deep belief networks.[8][12]

The term deep learning was introduced to the machine learning community by Rina Dechter in 1986,[13] and to artificial neural networks by Igor Aizenberg and colleagues in 2000, in the context of Boolean threshold neurons.[14][15] The etymology of the term is more complicated.[16]

''' 


result=chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()