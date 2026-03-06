$json="{""model"":""llama3"",""prompt"":""%%PROMPT%%"",""stream"":false}"

$params=@{
	Uri     = 'http://127.0.0.1:11434/api/generate'
	Method  = 'POST'
	Headers = @{
		"Content-Type"="application/json"
        "charset"="utf-8"
	}
	Body    = $json
}

Invoke-RestMethod @params

# example using enconding

#powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[Console]::OutputEncoding = [System.Text
# .Encoding]::UTF8; .\ollama.ps1"

# example response

#model                : llama3
#created_at           : 2026-03-02T22:07:47.8196328Z
#response             : ¡Buenos días! ¿Cómo estás hoy?
#done                 : True
#done_reason          : stop
#context              : {128006, 882, 128007, 271...}
#total_duration       : 95808871500
#load_duration        : 36328991100
#prompt_eval_count    : 14
#prompt_eval_duration : 38869268900
#eval_count           : 13
#eval_duration        : 20547073500

