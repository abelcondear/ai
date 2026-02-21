package com.ollama.ollama.configuration;

import java.util.List;

public class TaskPrompt {
    private String prompt;
    private List<String> response;

    public String getPrompt(){
        return this.prompt;
    }

    public void setPrompt(String prompt){
        this.prompt = prompt;
    }

    public List<String> getResponse(){
        return this.response;
    }

    public void setResponse(List<String> response){
        this.response = response;
    }
}
