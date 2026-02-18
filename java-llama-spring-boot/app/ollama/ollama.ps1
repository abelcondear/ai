param (
    [string]$Prompt
)

Invoke-RestMethod -Uri http://127.0.0.1:11434/api/generate -Method POST -Headers @{"Content-Type"="application/json"} -Body "{""model"":""llama3"",""prompt"":""$Prompt"",""stream"":false}"




