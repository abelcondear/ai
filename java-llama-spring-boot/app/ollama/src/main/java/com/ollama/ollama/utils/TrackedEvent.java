package com.ollama.ollama.utils;

import com.ollama.ollama.configuration.TaskPrompt;
import org.springframework.context.ApplicationEvent;

public class TrackedEvent extends ApplicationEvent {
    private final String taskId;
    private final TaskPrompt taskPrompt;

    public TrackedEvent(Object source, String taskId, TaskPrompt taskPrompt) {
        super(source);
        this.taskId = taskId;
        this.taskPrompt = taskPrompt;
    }

    public String getTaskId() {
        return this.taskId;
    }

    public TaskPrompt getTaskPrompt() {
        return this.taskPrompt;
    }
}