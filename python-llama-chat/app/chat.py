import os
from ollama import Client

client = Client\
(
    host='http://127.0.0.1:11434',
    headers=\
    {
      #'Authorization': 'Bearer AA...',
      #'Content-Type': 'application/x-ndjson'
    }    
)

messages =\
[
  {
    'role': 'system',
    'content': '''
You are an assistant for helping to find any locations. 

Other questions, say kindly that it is not part of 
your role using your own words. 

Be short and concise in your answers.
''',
  },
]

max_iteration = 2
counter = 0

while (True):
  userPrompt = input("Write any question you want to ask: ")

  if userPrompt.strip() == "":
    continue

  if userPrompt.strip().lower() == "bye":
    break

  messages.append\
  (
      {
          'role': 'user',
          'content': userPrompt,
      },
  )

  first_printing = True

  for part in client.chat('llama3', messages=messages, stream=True):
    if first_printing:
      print("AI (Llama): ", end="", flush=True)
      first_printing = False

    print(part.message.content, end="", flush=True)

  print("")
  print("")
  
  counter += 1

  if max_iteration == counter:
    break

