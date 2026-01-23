import subprocess
import random

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class SaveLogging:
    def __init__(self):
       pass 
    
    def run(self, value):
      command = ["python", "write.file.py", f"\"{value}\""]
      
      # execute process in the background
      process = subprocess.Popen(command)
      process.wait()

class Runnable:
    def __init__(self):
       pass 
    
    def run(self, id, value, priority):
       command = ["python", "connect.db.py", f"{id}", f"\"{value}\"", f"{priority}"]
       
       # execute process in the background
       process = subprocess.Popen(command)
       process.wait()

class MessageState(TypedDict):
   content: str
   requires_priority: bool
   command: Runnable
   log: SaveLogging

def parse_message(state: MessageState) -> MessageState:
   print("Analyzing message ...")

   keywords = [
      "as soon as possible",
      "need to buy"
   ]

   for kw in keywords:
      if kw.lower() in state["content"].lower():      
         # save message into database
         state["command"].run(random.randint(1, 9999), state["content"], 1)
         state["requires_priority"] = True
         break

   counter = 0
   for kw in keywords:
      if not kw.lower() in state["content"].lower():      
         counter += 1   
        
   if counter == len(keywords):
      # save message into database     
      state["command"].run(random.randint(1, 9999), state["content"], 0)
      state["requires_priority"] = False

   return state

def prioritize_message(state: MessageState) -> MessageState:
   if state["requires_priority"] == True:
       print("Logging prioritized message ...")
       state["log"].run(state["content"])

   return state

message_list = [
   "There is an issue with the printer. It needs to be repaired as soon as possible.",
   "One more desk should be added to the office when it is possible.",
   "Need to buy new cleaning products for the establishment."
]

for item in message_list:
   # Create workflow 
   workflow = StateGraph(MessageState)

   # Add nodes
   workflow.add_node("parse_message", parse_message)
   workflow.add_node("prioritize_message", prioritize_message)

   # Add edges
   workflow.add_edge(START, "parse_message")
   workflow.add_edge("parse_message", "prioritize_message")
   workflow.add_edge("prioritize_message", END)

   # Create custom objects
   CmdRunnable = Runnable()
   CmdLog = SaveLogging()

   message_graph = workflow.compile()

   final_state = message_graph.invoke({
      "content": item, 
      "requires_priority": False, 
      "command": CmdRunnable, 
      "log": CmdLog
   })

print("Done.")
print("")