#---------- -

param(
    [Parameter(Mandatory=$true)]
    [string]$Z
)

#Invoke-RestMethod -Uri "http://localhost:8080/home"

#---------- -

$chars=@{quote='"';singleQuote='''';equal='=';colon=':';braceOpen="{";braceClose="}";comma=',';dolar='$';semicolon=';';at='@';period='.';hyphen='-'}

#---------- -

#---------- -

$cmds=@{writeOutput='Write-Output'}

#---------- -

$uri=@{name='Uri';value='http://127.0.0.1:11434/api/generate'}

#---------- -

$uriVariable="{0}{1}{2}{3}{4}" -f $uri.name, $chars.equal, $chars.singleQuote, $uri.value, $chars.singleQuote

#---------- -

$method=@{name='Method';value='POST'}

#---------- -

$methodVariable="{0}{1}{2}{3}{4}" -f $method.name, $chars.equal, $chars.singleQuote, $method.value, $chars.singleQuote

#---------- -

$contentType=@{name='Content-Type';value='application/json'}

#---------- -

$contentTypeVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}" -f $chars.quote, $chars.quote, $chars.quote, $contentType.name, $chars.quote, $chars.quote, $chars.quote, $chars.equal, $chars.singleQuote, $contentType.value, $chars.singleQuote

#---------- -

$charSet=@{name='Charset';value='utf-8'}

#---------- -

$charSetVariable="{0}{1}{2}{3}{4}" -f $charSet.name, $chars.equal, $chars.singleQuote, $charSet.value, $chars.singleQuote

#---------- -

#------ json key-value

$model=@{name='model';value='llama3'}

#$prompt=@{name='prompt';value='%%PROMPT%%'}

#$promptParameterVariable='{0}{1}{2}{3}{4}{5}{6}' -f $chars.singleQuote,
#
#$chars.singleQuote,
#$chars.singleQuote,
#
#$Prompt,
#
#$chars.singleQuote,
#$chars.singleQuote,
#
#$chars.singleQuote

#$prompt=@{name='prompt';value='Hello'}

$prompt=@{name='prompt';value=$Z}
#$prompt=@{name='prompt';value=$promptParameterVariable}

$stream=@{name='stream';value='false'}

#---------- -

$modelVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}" -f $chars.quote, 

$chars.quote, 
$chars.quote, 

$model.name, 

$chars.quote, 
$chars.quote, 

$chars.quote, 

$chars.colon, 

$chars.quote, 

$chars.quote, 
$chars.quote, 

$model.value,

$chars.quote,
$chars.quote,

$chars.quote

# ---------- -


$promptVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}" -f $chars.quote,

$chars.quote,
$chars.quote,

$prompt.name, 

$chars.quote,
$chars.quote,

$chars.quote,

$chars.colon, 

$chars.quote,

$chars.quote,
$chars.quote,

$prompt.value, 

$chars.quote,
$chars.quote,

$chars.quote

#---------- -

$streamVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}" -f $chars.quote,

$chars.quote,
$chars.quote,

$stream.name, 

$chars.quote,
$chars.quote,

$chars.quote,

$chars.colon, 

$stream.value

#---------- -

$body=@{name='Body'}

$bodyVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}" -f $body.name, $chars.equal, $chars.singleQuote, $chars.braceOpen, $modelVariable, $chars.comma, $promptVariable, $chars.comma, $streamVariable, $chars.braceClose, $chars.singleQuote

#---------- -

$params=@{name='params'}

#---------- -

$invoke=@{command="Invoke-RestMethod";uriName="Uri";uriValue="params.Uri";methodName="Method";methodValue="params.Method";headersName="Headers";headersValue="params.Headers";bodyName="Body";bodyValue="params.Body"}

#---------- -

$commandInvoke="{0} {1}{2} {3}{4} {5}{6} {7}{8} {9}{10} {11}{12} {13}{14} {15}{16}" -f $invoke.command, $chars.hyphen, $invoke.uriName, $chars.dolar, $invoke.uriValue, $chars.hyphen, $invoke.methodName, $chars.dolar, $invoke.methodValue, $chars.hyphen, $invoke.headersName, $chars.dolar, $invoke.headersValue, $chars.hyphen, $invoke.bodyName, $chars.dolar, $invoke.bodyValue

#---------- -

$headersKeyValueVariable="{0}{1}{2};{3};{4}" -f $chars.at, $chars.braceOpen, $contentTypeVariable, $charSetVariable, $chars.braceClose

$headers=@{name='Headers';value=$headersKeyValueVariable}

#---------- -

$headersVariable="{0}{1}{2}" -f  $headers.name, $chars.equal, $headers.value

#--- -----

#$command="{0}{1}{2}{3}{4}{5};{6};{7};{8}{9};{10}{11};{12}" -f $chars.dolar, $params.name, $chars.equal, $chars
# .at, $chars.braceOpen, $uriVariable, $methodVariable, $headersVariable, $bodyVariable, $chars.braceClose, $chars.dolar, $params.name, $commandInvoke

$command="{0}{1}{2}{3}{4}{5};{6};{7};{8}{9};{10}" -f $chars.dolar, $params.name, $chars.equal, $chars.at, $chars.braceOpen, $uriVariable, $methodVariable, $headersVariable, $bodyVariable, $chars.braceClose, $commandInvoke

#Write-Output $command
Powershell -Command $command

#--- --


# ------------------
# Example Response
# ------------------

# Name                           Value
# ----                           -----
# Method                         POST
# Body                           {"model":"llama3","prompt":"hello","stream":false}
# Headers                        {Charset, Content-Type}
# Uri                            http://127.0.0.1:11434/api/generate

# model                : llama3
# created_at           : 2026-03-16T02:30:40.5615084Z
# response             : Hello! It's nice to meet you. Is there something I can help you with, or would you
#                        like to chat?
# done                 : True
# done_reason          : stop
# context              : {128006, 882, 128007, 271...}
# total_duration       : 107539092400
# load_duration        : 27982438500
# prompt_eval_count    : 11
# prompt_eval_duration : 27274012100
# eval_count           : 26
# eval_duration        : 52159968700
