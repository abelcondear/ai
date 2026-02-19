package com.ollama.ollama.configuration;

public class TaskPrompt {
    private String prompt;
    private String response;

    public String getPrompt(){
        return this.prompt;
    }

    public void setPrompt(String prompt){
        this.prompt = prompt;
    }

    public String getResponse(){
        return this.response;
    }

    public void setResponse(String response){
        this.response = response;
    }
}
