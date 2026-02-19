package com.ollama.ollama.component;

import com.ollama.ollama.configuration.TaskPrompt;
import org.springframework.stereotype.Component;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@Component
public class TaskPromptStore {
    // create array of promptMap
    private final ConcurrentMap<String, TaskPrompt> promptMap = new ConcurrentHashMap<>();

    public void setPrompt(String taskId, TaskPrompt prompt) {
        promptMap.put(taskId, prompt);
    }

    public TaskPrompt getPrompt(String taskId) {
        TaskPrompt prompt = new TaskPrompt(); // identify the right prompt using taskId
        return promptMap.getOrDefault(taskId, prompt);
    }
}