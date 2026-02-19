package com.ollama.ollama.service;

import com.ollama.ollama.component.TaskPromptStore;
import com.ollama.ollama.configuration.TaskPrompt;
import com.ollama.ollama.configuration.TaskStatus;
import com.ollama.ollama.component.TaskStatusStore;
import com.ollama.ollama.utils.TrackedEvent;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import java.util.UUID;

@Service
public class TrackedEventService {

    private final ApplicationEventPublisher publisher;
    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    public TrackedEventService(ApplicationEventPublisher publisher, TaskStatusStore statusStore,
                               TaskPromptStore promptStore) {
        this.publisher = publisher;
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    public String triggerTrackedEvent(String prompt) {
        String taskId = UUID.randomUUID().toString();
        TaskPrompt taskPrompt = new TaskPrompt();

        taskPrompt.setPrompt(prompt);
        promptStore.setPrompt(taskId, taskPrompt);
        statusStore.setStatus(taskId, TaskStatus.PENDING);
        TrackedEvent trackedEvent = new TrackedEvent(this, taskId, taskPrompt);

        publisher.publishEvent(trackedEvent);
        return taskId;
    }
}