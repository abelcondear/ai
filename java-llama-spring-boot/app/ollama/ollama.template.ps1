#---------- -

param(
    [Parameter(Mandatory=$true)]
    [string]$Z
)

#Invoke-RestMethod -Uri "http://localhost:8080/home"

#---------- -

$chars=@{quote='"';singleQuote='''';equal='=';colon=':';braceOpen="{";braceClose="}";comma=',';dolar='$';
semicolon=';';at='@';period='.';hyphen='-';underscore='_';parenthesisOpen="(";parenthesisClose=")";}

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

#--- -----

Powershell -Command $command
Exit

#--- -----

$tryBegin="try {0}" -f $chars.braceOpen
#Write-Output $tryBegin
#Exit

#$tryEnd="{0} catch {1}
#    Write-Output {2}{3}An error occurred{4}{5}{6}
#    Write-Output {7}{8}{9}Exception{10}Message
#
#    {11}errorCode {12} {13}1
#{14}
#
#    #if {14}{15}{16}{17}Exception {18}and {19}{20}{21}Exception{22}HResult{23} {24}
#    #    {25}errorCode {26} {27}{28}{29}Exception{30}HResult
#    #{31}
#    #elseif {32}{33}{34}{35}Exception {36}and {37}{38}{39}Exception{40}InnerException {41}and
#    #{42}{43}{44}Exception{45}InnerException{46}HResult{47} {48}
#    #    {49}errorCode {50} {51}{52}{53}Exception{54}InnerException{55}HResult
#    #{56}
#
#    #Write-Output {57}errorCode
##{58}
#" -f $chars.braceClose,

$tryEnd="{0} catch {1}
    Write-Output {2}{3}An error occurred{4}{5}{6}
    Write-Output {7}{8}{9}Exception{10}Message

    {11}errorCode {12} {13}1
{14}
" -f $chars.braceClose,

#catch {1}
#Write-Output {2}{3}An error occurred{4}{5}{6}
#Write-Output {7}{8}{00}Exception{9}Message
$chars.braceOpen, $chars.quote,
$chars.quote, $chars.colon, $chars.quote, $chars.quote, $chars.dolar, $chars.underscore, $chars.period, $chars.period,

#{10}errorCode {11} {12}1
#$chars.dolar, $chars.equal, $chars.hyphen

# overwrite command
$chars.dolar, $chars.equal, $chars.hyphen, $chars.braceClose

Write-Output $tryEnd
Exit

#if {13}{14}{15}{16}Exception {17}and {18}{19}{20}Exception{21}HResult{22} {23}
#{24}errorCode {25} {26}{27}{28}Exception{29}HResult
#{30}
#$chars.parenthesisOpen, $chars.dolar, $chars.underscore, $chars.period, $chars.hyphen, $chars.dolar, $chars
# .underscore, $chars.period, $char.period, $chars.parenthesisClose, $chars.braceOpen, $chars.dolar, $chars.equal, $chars.dolar, $chars.underscore, $chars.period, $chars.period, $chars.braceClose,

#elseif {31}{32}{33}{34}Exception {35}and {36}{37}{38}Exception{39}InnerException {40}and{41}{42}{43}Exception{44}InnerException{45}HResult{46} {47}

#$chars.parenthesisOpen, $chars.dolar, $chars.underscore, $chars.period, $chars.hyphen, $chars.dolar, $chars
# .underscore, $chars.period, $chars.period, $chars.hyphen, $chars.dolar, $chars.underscore, $chars.period, $chars.period, $chars.period, $chars.parenthesisClose, $chars.braceOpen,

#   {48}errorCode {49} {50}{51}{52}Exception{53}InnerException{54}HResult
#{55}
#$chars.dolar, $chars.equal, $chars.dolar, $chars.underscore, $chars.period, $chars.period, $chars.period,
# $chars.braceClose,

#   Write-Output {56}errorCode
#{57}
#$chars.dolar,
#$chars.braceClose

# overwrite command
$command="{0}{1}{2}{3}{4}{5};{6};{7};{8}{9};{10};" -f $chars.dolar, $params.name, $chars.equal, $chars.at, $chars.braceOpen, $uriVariable, $methodVariable, $headersVariable, $bodyVariable, $chars.braceClose, $commandInvoke

$command="    {0}" -f $command

$command="{0}
    {1}
{2}" -f $tryBegin, $command, $tryEnd

Write-Output $command

#Write-Output "------"
#Write-Output $tryBegin
#Write-Output $command
##Write-Output "------"
#Write-Output $tryEnd
#Write-Output "------"
#Exit

# try-catch -- should be inside $command (string)
#Powershell -Command $command

#--- --

#try {
#    # Force an error if the file doesn't exist
#    $content = Get-Content -Path "C:\nonexistent\file.txt" -ErrorAction Stop
#    Write-Host "File content loaded successfully."
#}
#catch {
#    # $_ contains the current error object
#    Write-Host "An error occurred:" -ForegroundColor Red
#    Write-Host $_.Exception.Message -ForegroundColor Yellow  # Short message
#    Write-Host "Full error details:" -ForegroundColor Cyan
#    Write-Host $_.ToString()                                 # Full error text
#}
#finally {
#    Write-Host "Execution finished." -ForegroundColor Green
#}

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
