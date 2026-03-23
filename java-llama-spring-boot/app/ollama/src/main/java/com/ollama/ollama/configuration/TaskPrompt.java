package com.ollama.ollama.configuration;

import java.util.ArrayList;
import java.util.List;

public class TaskPrompt {
    private String prompt;
    private List<String> response = new ArrayList<>(){};
    private List<String> error = new ArrayList<>(){};

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

    public List<String> getError(){
        return this.error;
    }

    public void setError(List<String> error){
        this.error = error;
    }
}
