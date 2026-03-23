package com.ollama.ollama.service;

import com.ollama.ollama.component.TaskPromptStore;
import com.ollama.ollama.configuration.TaskPrompt;
import com.ollama.ollama.configuration.TaskStatus;
import com.ollama.ollama.component.TaskStatusStore;
import com.ollama.ollama.model.TrackedEvent;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import java.util.UUID;

@Service
public class TrackedEventService {

    private final ApplicationEventPublisher publisher;
    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    public String taskId;
    public TaskPrompt taskPrompt;

    public TrackedEventService(ApplicationEventPublisher publisher, TaskStatusStore statusStore,
                               TaskPromptStore promptStore) {
        this.publisher = publisher;
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    public String triggerTrackedEvent(String prompt) {
        this.taskId = UUID.randomUUID().toString();
        //String taskId = UUID.randomUUID().toString();
        TaskPrompt taskPrompt = new TaskPrompt();

        taskPrompt.setPrompt(prompt);
        promptStore.addPrompt(this.taskId, taskPrompt);
        statusStore.addStatus(this.taskId, TaskStatus.PENDING);
        TrackedEvent trackedEvent = new TrackedEvent(this, this.taskId, taskPrompt);

        this.taskPrompt = promptStore.getPrompt(this.taskId);

        publisher.publishEvent(trackedEvent);
        return taskId;
    }
}