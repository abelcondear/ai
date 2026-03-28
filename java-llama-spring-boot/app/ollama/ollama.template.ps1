#---------- -

param(
    [Parameter(Mandatory=$true)]
    [string]$UserPrompt
)

#---------- -

#$postParams = @{editPromptES='Hola'}

#Invoke-WebRequest -Uri http://localhost:8080/home -Method POST -Body $postParams

#StatusCode        : 200
#StatusDescription :
#Content           : <!DOCTYPE html>
#<html>
#<head>
#<meta charset="UTF-8">
#<title>ollama</title>
#<style>
#body { font-family: Arial, sans-serif;
#text-align: center; padding: 50px; }
#</style>
#...
#RawContent        : HTTP/1.1 200
#Content-Language: es-ES
#Transfer-Encoding: chunked
#Content-Type: text/html;charset=UTF-8
#Date: Wed, 25 Mar 2026 14:44:38 GMT
#
#<!DOCTYPE html>
#<html>
#<head>
#<meta charset="UT...
#Forms             : {}
#Headers           : {[Content-Language, es-ES], [Transfer-Encoding,
#                    chunked], [Content-Type,
#                    text/html;charset=UTF-8], [Date, Wed, 25 Mar
#                    2026 14:44:38 GMT]}
#Images            : {}
#InputFields       : {@{innerHTML=; innerText=; outerHTML=<INPUT
#                    name=editPromptES placeholder="Escriba su
#consulta">; outerText=; tagName=INPUT;
#                    name=editPromptES; placeholder=Escriba su
#                    consulta}, @{innerHTML=; innerText=;
#                    outerHTML=<INPUT type=submit value=Go>;
#                    outerText=; tagName=INPUT; type=submit;
#                    value=Go}}
#Links             : {}
#ParsedHtml        : mshtml.HTMLDocumentClass
#RawContentLength  : 1175


#---------- -

$chars=@{quote='"';singleQuote='''';equal='=';colon=':';braceOpen="{";braceClose="}";comma=',';dolar='$'; semicolon=';';at='@';period='.';hyphen='-';underscore='_';parenthesisOpen="(";parenthesisClose=")"; squareBracketsOpen="[";squareBracketsClose="]";lessThan="<";greaterThan=">";}

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

$contentTypeVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}" -f $chars.quote,

$chars.quote,
$chars.quote,

$contentType.name,

$chars.quote,
$chars.quote,

$chars.quote,

$chars.equal,

$chars.singleQuote,
$contentType.value,
$chars.singleQuote

#---------- -

$charSet=@{name='Charset';value='utf-8'}

#---------- -

$charSetVariable="{0}{1}{2}{3}{4}" -f $charSet.name,

$chars.equal,

$chars.singleQuote,
$charSet.value,
$chars.singleQuote

#---------- -

#------ json key-value

$model=@{name='model';value='llama3'}

$prompt=@{name='prompt';value=$UserPrompt}

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

$bodyVariable="{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}" -f $body.name,

$chars.equal,

$chars.singleQuote,

$chars.braceOpen,

$modelVariable,
$chars.comma,

$promptVariable,
$chars.comma,

$streamVariable,

$chars.braceClose,

$chars.singleQuote

#---------- -

$params=@{name='params'}

#---------- -

$invoke=@{command="Invoke-RestMethod";uriName="Uri";uriValue="params.Uri";methodName="Method";methodValue="params.Method";headersName="Headers";headersValue="params.Headers";bodyName="Body";bodyValue="params.Body"}

#---------- -

$commandInvoke="{0} {1}{2} {3}{4} {5}{6} {7}{8} {9}{10} {11}{12} {13}{14} {15}{16}" -f $invoke.command,

$chars.hyphen,
$invoke.uriName,

$chars.dolar,
$invoke.uriValue,

$chars.hyphen,
$invoke.methodName,

$chars.dolar,
$invoke.methodValue,

$chars.hyphen,
$invoke.headersName,

$chars.dolar,
$invoke.headersValue,

$chars.hyphen,
$invoke.bodyName,

$chars.dolar,
$invoke.bodyValue

#---------- -

$headersKeyValueVariable="{0}{1}{2};{3};{4}" -f $chars.at,

$chars.braceOpen,

$contentTypeVariable,
$charSetVariable,

$chars.braceClose

$headers=@{name='Headers';value=$headersKeyValueVariable}

#---------- -

$headersVariable="{0}{1}{2}" -f  $headers.name, $chars.equal, $headers.value

#--- -----

$command="{0}{1}{2}{3}{4}{5};{6};{7};{8}{9};{10}" -f $chars.dolar,

$params.name,

$chars.equal,

$chars.at,
$chars.braceOpen,

$uriVariable,

$methodVariable,

$headersVariable,

$bodyVariable,

$chars.braceClose,

$commandInvoke

# overwrite variable to test -- ran successfully
# overwrite variable to test -- ran successfully -- try II
$command="{0}{1}{2}{3}{4}{5};{6};{7};{8}{9};

#{10}{11}; # debug # disabled

{12};
" -f $chars.dolar,

$params.name,

$chars.equal,

$chars.at,

$chars.braceOpen,

$uriVariable,
$methodVariable,
$headersVariable,
$bodyVariable,

$chars.braceClose,

$chars.dolar, # debug purpose
$params.name, # debug purpose

$commandInvoke


#--- -----

$tryBegin="try {0}" -f $chars.braceOpen

#--- -----

#--- -----

#    Write-Output {2}{3}{4}An error occurred{5}{6}{7}{8};
#
#    {9}errorMessage{10}{11}{12}{13}Exception{14}Message;
#
#    throw {15}System.Exception{16}{17}{18}new{19} {20}errorMessage{21};
#
#    {22}errorCode {23} {24}1;

#throw {8}System.Exception{9}{10}{11}new{12} {13}errorMessage{14};

#$tryEnd="{0} catch {1}
#    {2}errorMessage{3}{4}{5}{6}Exception{7}Message;

#throw {8}System.Exception{9}{10}{11}new{12}{13}errorMessage{14};
#{15}
#" -f $chars.braceClose,


$tryEnd="{0} catch {1}
    {2}errorMessage{3}{4}String{5}{6}{7}Concat{8}
    {9}{10}{11}{12}{13}{14}{15}{16}{17},
    {18}{19}{20}Exception{21}Message,
    {22}{23}{24}{25}{26}{27}{28}{29}{30}
    {31};

    throw {32}System.Exception{33}{34}{35}new{36}{37}errorMessage{38};
{39}
" -f $chars.braceClose,

#[string]::Concat($string1, $string2)


#catch {1}
#Write-Output {2}{3}An error occurred{4}{5}{6}
#Write-Output {7}{8}{00}Exception{9}Message


#{0} catch {1}
$chars.braceOpen,

###old one. discarded
###{2}errorMessage{3}{4}{5}{6}Exception{7}Message;

#{2}errorMessage{3}{4}{5}{6}{7}Concat{8}
#{9}{10}{11}{12}{13}{14}{15}{16}{17},
#{18}{19}{20}Exception{21}Message,
#{22}{23}{24}{25}{26}{27}{28}{29}{30},
#{31};

$chars.dolar,
$chars.equal,

$chars.squareBracketsOpen,
$chars.squareBracketsClose,

$chars.colon,
$chars.colon,

$chars.parenthesisOpen,

$chars.quote,
$chars.quote,
$chars.quote,

$chars.lessThan,
$chars.lessThan,
$chars.lessThan,

$chars.quote,
$chars.quote,
$chars.quote,

$chars.dolar,
$chars.underscore,
$chars.period,
$chars.period,

$chars.quote,
$chars.quote,
$chars.quote,

$chars.greaterThan,
$chars.greaterThan,
$chars.greaterThan,

$chars.quote,
$chars.quote,
$chars.quote,

$chars.parenthesisClose,


# ommitted
#throw [System.Exception]::new("Custom error: Value cannot be zero.")

# valid
#throw [System.Exception]::new("Custom error: Value cannot be zero.")

# using index parameters
#throw {13}System.Exception{14}{15}{16}new({17}{18}{19} {00000} {20} {21}{19}{20}{21})

# deprecated
#throw {8}System.Exception{9}{10}{11}new{12}{13}errorMessage{14};

#throw {32}System.Exception{33}{34}{35}new{36}{37}errorMessage{38};
$chars.squareBracketsOpen,
$chars.squareBracketsClose,

$chars.colon,
$chars.colon,

$chars.parenthesisOpen,

$chars.dolar,

$chars.parenthesisClose,

###{15}
#{39}
$chars.braceClose

#Write-Output $tryEnd
#Exit

#--- -----

$command = $command.replace("`r`n","`r`n    ")

#--- -----

$command="{0}
    {1}
{2}" -f $tryBegin, $command , $tryEnd

Write-Output $command
Powershell -Command $command

#--- -----

# ------------------
# Example Response
# ------------------

# POST Method

# Name                           Value
# ----                           -----
# Method                         POST
# Body                           {"model":"llama3","prompt":"hello","stream":false}
# Headers                        {Charset, Content-Type}
# Uri                            http://127.0.0.1:11434/api/generate

# -------

# API Result

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

# -------
