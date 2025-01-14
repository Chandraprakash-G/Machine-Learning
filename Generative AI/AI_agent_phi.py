from phi.agent import Agent
import os
from phi.model.groq import Groq
from dotenv import load_dotenv
load_dotenv()

print(os.environ['GROQ_API_KEY'])

agent = Agent(provider=Groq(id='llama-3.3-70b-versatile'))
agent.print_response('What is Linear Regression')
