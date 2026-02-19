package com.ollama.ollama.controller;

import com.ollama.ollama.component.TaskPromptStore;
import com.ollama.ollama.component.TaskStatusStore;
import com.ollama.ollama.service.TrackedEventService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TrackedEventController {
    private final TrackedEventService service;
    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    public TrackedEventController(TrackedEventService service, TaskStatusStore statusStore,
                                  TaskPromptStore promptStore) {
        this.service = service;
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    @GetMapping("/trigger-tracked/{prompt}")
    public String triggerTracked(@PathVariable("prompt") String prompt) {
        String taskId = service.triggerTrackedEvent(prompt);
        return "Task started. ID: " + taskId + " — check status at /status/" + statusStore.getStatus(taskId);
    }

    @GetMapping("/trigger-response/{taskId}")
    public String triggerResponse(@PathVariable("taskId") String taskId) {
        return "Task " + taskId + " - prompt: " + promptStore.getPrompt(taskId).getPrompt() + " - response: " + promptStore.getPrompt(taskId).getResponse();
    }

    @GetMapping("/status/{taskId}")
    public String getStatus(@PathVariable("taskId") String taskId) {
        return "Task " + taskId + " status: " + statusStore.getStatus(taskId);
    }
}