import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# open-source model
model_id = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer= AutoTokenizer.from_pretrained(model_id)
gen_pipe = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	max_new_tokens=512,
	do_sample=True,
	temperature=0.7,
	top_p=0.9
)
llm = HuggingFacePipeline(pipeline=gen_pipe)

# FAISS setup
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("data/vectorstore", embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 5})

# Web search tool
tavily = TavilySearchResults(api_key="your_tavily_api_key")

# Agents Definitions
def preference_analyst_agent():
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""You are a travel analyst. Based on the user preferences below, 
            generate a concise search query to look for destinations: {user_input}
        """
    )
    return LLMChain(llm=llm,prompt=prompt)

def recommender_agent(label: str):
    prompt = PromptTemplate(
        input_variables=["query", "context", "web"],
        template=f"""
		You are Proposer {label}. Propose one travel destination that fits:

		Query: {{query}}

		Database Context:
		{{context}}

		Live Web Info:
		{{web}}

		Suggest a destination and briefly describe it."""
    )
    return LLMChain(llm=llm, prompt=prompt)

# Refiner (enhance an existing option)
def refiner_agent(label):
    prompt = PromptTemplate(
        input_variables=["proposal", "critique", "fc_critique" "web"],
        template=f"""
			You are Refiner {label}. Improve your existing proposal based on this critique and any new web facts:
			- Previous proposal: {{proposal}}
			- Critique received: {{critique}}
			- Fact checks on critique received: {{fc_critique}}
			- Web evidence: {{web}}
			Return an enhanced proposal only.
		"""
    )
    return LLMChain(llm=llm, prompt=prompt)

# Challenger (Critique) with readiness flag
def critic_agent(label):
    prompt = PromptTemplate(
        input_variables=["own_proposal", "opponent_proposal", "fc_opponent_proposal" "web"],
        template=f"""
		You are Challenger {label}. 
        You proposed: {{own_proposal}}
        Opponent: {{opponent_proposal}}
        Web info: {{web}}
        Fact checks on opponent's proposal: {{fc_opponent_proposal}}
		Write your critique.
	"""
    )
    return LLMChain(llm=llm, prompt=prompt)

def manager_agent(label):
    prompt = PromptTemplate(
        input_variables=["own_proposal", "opponent_proposal", "own_critique", "opponent_critique"],
        template= f"""
			You are Manager{label}
			You proposed: {{own_proposal}}
            Opponennt: {{opponent_proposal}}
            Your critique on opponent's proposal: {{own_critique}}
            Opponent critique on your proposal: {{opponent_critique}}
            Return [READY] if you believe the evaluator will favor your proposal. Return [More] if you don't believe you will win, and would like another round of debate.
        """
	)
   
def fact_checker_agent():
    prompt = PromptTemplate(
        input_variables=["claim", "web"],
        template = """
			You are a fact checker. Given the proposal or critique below, identify any inaccuracies or unsupported statements and provide corrected evidence with the help of web results if needed.
            Claim: {claim}
            Web Info: {web}
		"""
	)

def evaluator_agent():
    prompt = PromptTemplate(
        input_variables=["proposal_a", "proposal_b","critique_a","critique_b"],
        template = """Evaluate the two proposals and their critiques:
            Proposal A:
            {proposal_a}
            Critique A:
            {critique_a}
            
            Proposal B:
            {proposal_b}
            Critique B:
            {critique_b}

            Choose the better one or suggest a blend. Justify briefly.
        """
    )
    return LLMChain(llm=llm, prompt=prompt)

def feedback_agent():
    prompt=PromptTemplate(
        input_variables=["recommendation"],
        template="""
          Here is your travel recommendation:
          {recommendation}

          Are you satisfied? Say "stop" or provide refined preferences to try again.
        """
    )
    return LLMChain(llm=llm, prompt=prompt)
