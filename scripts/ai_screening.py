import os
import pandas as pd
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import Literal, List, Optional


class OutputSchema(BaseModel):
    thoughts: str = Field(description= "thoughts of the model")
    decision: int = Field(description="1 if the article is SELECTED as it falls with the scope of the REVIEW_TOPIC, 0 otherwise")
    reason: str = Field(description = "generate a consise one sentence long reason for the decision")
    ai_method_list: Optional[List[str]] = Field(default=None, description="list all the AI method explored in the article")


def main():

  llm = ChatOllama(model="deepseek-r1:32b", temperature=0.0, num_ctx=5_000)
  struct_llm = llm.with_structured_output(OutputSchema)
  
  df = pd.read_csv(os.path.join("..", "dataset/screening/articles_to_screen.csv"))[["key", "title", "abstract"]]
  out_dir = "results"

  if not os.path.exists(out_dir):
     os.makedirs(out_dir)
  
  messages = [
    ('system',  "You are a helpful AI reviewer that ACCURATELY SCREENS and SELECTS 'ORIGINAL RESEARCH ARTICLES' that falls within the scope of the given 'LITERATURE_REVIEW_TOPIC', based on their ABSTRACT." 
                "Your decision should be '1' if SELECTED or '0' otherwise."
                "Generate a concise, one-sentence reason to motivate your decision."
                "If selected, list all the AI methods explored in the ORIGINAL RESEARCH ARTICLE."
    ),
    ('human', "ABSTRACT:\n\n title: {title}, \n content: {abstract}\n\n LITERATURE_REVIEW_TOPIC: APPLICATIONS OF AI IN CARBON ION THERAPY")
  ]

  prompt_template = ChatPromptTemplate.from_messages(messages)
  chain = prompt_template | struct_llm
  
  decision_df = {"key":[], "title":[], "abstract":[], "decision":[], "ai_method_list":[], "reason":[], "thoughts":[]}
  
  rows = [row.to_dict() for i,row in df.iterrows()]
  
  for row in tqdm(rows):
      # key, title, selection = row.to_dict()["key"], row.to_dict()["title"], row.to_dict()["selection"]
  
      inputs = {key:row[key] for key in ["title", "abstract"]} 
  
      output = chain.invoke(inputs)
  
      for key,val in {**row, **dict(output)}.items():
          decision_df[key].append(val)
  
  decision_df = pd.DataFrame(decision_df)
  decision_df.to_csv(os.path.join(out_dir, "ai_decision_new.csv"), index=False)
  
if __name__=="__main__":
  main()
    

