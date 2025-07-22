import os
import pandas as pd
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field


class OutputSchema(BaseModel):
    thoughts: str = Field(description= "thoughts of the model")
    decision: int = Field(description="1 if the abstract was selected, 0 otherwise")
    reason: str = Field(description = "generate a consise one sentence long reason for the decision")


def main():

  llm = ChatOllama(model="deepseek-r1:32b", temperature=0.0)
  
  struct_llm = llm.with_structured_output(OutputSchema)
  
  df = pd.read_csv(os.path.join("..", "notebooks/articles.csv"))[["key", "title", "abstract"]]
  
  messages = [
    ('system',  "You are a helpful AI agent that assists in accurately screening the article using its abstract. " 
                "Based solely on the abstract provided, determine whether the article discusses an APPLICATION OF AI METHODS IN CARBON ION THERAPY? "
                "Your decision should be '0' for NO or '1' for YES. Then, generate a concise, one-sentence reason for your decision."
    ),
    ('human', "abstract:\n\n title: {title}, \n content: {abstract}")
  ]

  prompt_template = ChatPromptTemplate.from_messages(messages)
  chain = prompt_template | struct_llm
  
  decision_df = {"key":[], "title":[], "abstract":[], "decision":[], "reason":[], "thoughts":[]}
  
  rows = [row.to_dict() for i,row in df.iterrows()]
  
  for row in tqdm(rows):
      # key, title, selection = row.to_dict()["key"], row.to_dict()["title"], row.to_dict()["selection"]
  
      inputs = {key:row[key] for key in ["title", "abstract"]} 
  
      output = chain.invoke(inputs)
  
      for key,val in {**row, **dict(output)}.items():
          decision_df[key].append(val)
  
  decision_df = pd.DataFrame(decision_df)
  decision_df.to_csv("ai_screening_outputs_final.csv", index=False)
  
if __name__=="__main__":
  main()
    
